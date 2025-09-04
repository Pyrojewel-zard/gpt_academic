import json
import re
import os
import time
import glob
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Generator, Tuple
from crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive
from toolbox import update_ui, promote_file_to_downloadzone, write_history_to_file, CatchException, report_exception
from shared_utils.fastapi_server import validate_path_safety
from crazy_functions.paper_fns.paper_download import extract_paper_id, extract_paper_ids, get_arxiv_paper, format_arxiv_id
import difflib
import re


@dataclass
class PaperQuestion:
    """论文分析问题类"""
    id: str  # 问题ID
    question: str  # 问题内容
    importance: int  # 重要性 (1-5，5最高)
    description: str  # 问题描述


class BatchPaperAnalyzer:
    """批量论文快速分析器"""

    def __init__(self, llm_kwargs: Dict, plugin_kwargs: Dict, chatbot: List, history: List, system_prompt: str):
        """初始化分析器"""
        self.llm_kwargs = llm_kwargs
        self.plugin_kwargs = plugin_kwargs
        self.chatbot = chatbot
        self.history = history
        self.system_prompt = system_prompt
        self.paper_content = ""
        self.results = {}
        self.paper_file_path = None
        # ---------- 读取分类树 ----------
        json_path = os.path.join(os.path.dirname(__file__), 'paper.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            self.category_tree = json.load(f)          # Dict[str, List[str]]

        # 生成给 LLM 的当前分类清单
        category_lines = [f"{main} -> {', '.join(subs)}"
                        for main, subs in self.category_tree.items()]
        self.category_prompt_str = '\n'.join(category_lines)
        # 定义论文分析问题库（与Paper_Reading保持一致）
        self.questions = [
            PaperQuestion(
                id="research_and_methods",
                question="这篇论文的主要研究问题、目标和方法是什么？请分析：1)论文的核心研究问题和研究动机；2)论文提出的关键方法、模型或理论框架；3)这些方法如何解决研究问题。",
                importance=5,
                description="研究问题与方法"
            ),
            PaperQuestion(
                id="findings_and_innovation",
                question="论文的主要发现、结论及创新点是什么？请分析：1)论文的核心结果与主要发现；2)作者得出的关键结论；3)研究的创新点与对领域的贡献；4)与已有工作的区别。",
                importance=4,
                description="研究发现与创新"
            ),
            PaperQuestion(
                id="methodology_and_data",
                question="论文使用了什么研究方法和数据？请详细分析：1)研究设计与实验设置；2)数据收集方法与数据集特点；3)分析技术与评估方法；4)方法学上的合理性。",
                importance=3,
                description="研究方法与数据"
            ),
            PaperQuestion(
                id="limitations_and_impact",
                question="论文的局限性、未来方向及潜在影响是什么？请分析：1)研究的不足与限制因素；2)作者提出的未来研究方向；3)该研究对学术界和行业可能产生的影响；4)研究结果的适用范围与推广价值。",
                importance=2,
                description="局限性与影响"
            ),
            PaperQuestion(
                id="worth_reading_judgment",
                question="请综合评估这篇论文是否值得精读，并从多个角度给出判断依据：1) **创新性与重要性**：论文的研究是否具有开创性？是否解决了领域内的关键问题？2) **方法可靠性**：研究方法是否严谨、可靠？实验设计是否合理？3) **论述清晰度**：论文的写作风格、图表质量和逻辑结构是否清晰易懂？4) **潜在影响**：研究成果是否可能对学术界或工业界产生较大影响？5) **综合建议**：结合以上几点，给出“强烈推荐”、“推荐”、“一般”或“不推荐”的最终评级，并简要说明理由。",
                importance=2,
                description="是否值得精读"
            ),
            PaperQuestion(
                id="category_assignment",
                question=(
                    "请根据论文内容，判断其最准确的二级分类归属。\n\n"
                    "当前分类树如下（一级 -> 二级）：\n"
                    f"{self.category_prompt_str}\n\n"
                    "要求：\n"
                    "1) 若完全匹配现有二级分类，直接回答：\n"
                    "   归属：<一级类别> -> <二级子分类>\n"
                    "2) 若需新建二级分类，回答：\n"
                    "   新增二级：<一级类别> -> <新子分类名>\n"
                    "3) 若需新建一级类别，回答：\n"
                    "   新增一级：<新一级类别> -> [<子分类1>, <子分类2>, ...]\n"
                    "4) 用一句话说明判断理由。"
                ),
                importance=1,
                description="论文二级分类归属"
            ),              

            PaperQuestion(
                id="core_algorithm_flowcharts",
                question=(
                    "请基于论文内容，绘制论文核心算法或核心思路的流程图，若论文包含多个相对独立的模块或阶段，请分别给出多个流程图。\n\n"
                    "要求：\n"
                    "1) 每个流程图使用 Mermaid 语法，代码块需以 ```mermaid 开始，以 ``` 结束；\n"
                    "2) 推荐使用 flowchart TD 或 LR，节点需概括关键步骤/子模块，包含主要数据流与关键分支/判定；\n"
                    "3) 每个流程图前以一句话标明模块/阶段名称，例如：模块：训练阶段；\n"
                    "4) 仅聚焦核心逻辑，避免过度细节；\n"
                    "5) 若只有单一核心流程，仅输出一个流程图；\n"
                    "6) 格式约束：\n"
                    "   - 节点名用引号包裹，如 [\"节点名\"] 或 (\"节点名\")；\n"
                    "   - 箭头标签采用 |\"标签名\"| 形式，且 | 与 \" 之间不要有空格；\n"
                    "   - 根据逻辑选择 flowchart LR（从左到右）或 flowchart TD（从上到下）。\n"
                    "7) 示例：\n"
                    "```mermaid\n"
                    "flowchart LR\n"
                    "    A[\"输入\"] --> B(\"处理\")\n"
                    "    B --> C{\"是否满足条件\"}\n"
                    "    C --> D[\"输出1\"]\n"
                    "    C --> |\"否\"| E[\"输出2\"]\n"
                    "```"
                ),
                importance=5,
                description="核心算法/思路流程图（Mermaid）"
            ),
            PaperQuestion(
                id="core_idea_ppt_md",
                question=(
                    "请生成一份用于 PPT 的‘论文核心思路与算法’极简 Markdown 摘要，并与已生成的 Mermaid 流程图形成配套说明。\n\n"
                    "输出格式要求（严格遵守）：\n"
                    "# 总述（1 行）\n"
                    "- 用最简一句话概括论文做了什么、为何有效。\n\n"
                    "# 模块要点（与流程图对应）\n"
                    "- 若存在多个流程图/模块：按“模块：名称”分组，每组列出 3-5 条‘图解要点’，每条 ≤ 14 字，概括核心输入→处理→输出与关键分支。\n"
                    "- 若仅有一个流程图：仅输出该流程图的 3-5 条‘图解要点’。\n\n"
                    "# 关键算法摘要（5-8 条）\n"
                    "- 每条 ≤ 16 字，聚焦输入/步骤/输出/创新，不写背景。\n\n"
                    "# 应用与效果（≤ 3 条，可省略）\n"
                    "- 场景/指标/收益。\n\n"
                    "注意：仅输出上述 Markdown 结构，不嵌入代码，不重复流程图本身。"
                ),
                importance=5,
                description="PPT 用核心思路与算法（Markdown 极简版）"
            ),
        ]

        # 按重要性排序
        self.questions.sort(key=lambda q: q.importance, reverse=True)

    # ---------- 关键词库工具 ----------
    def _get_keywords_db_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), 'keywords.txt')

    def _load_keywords_db(self) -> List[str]:
        path = self._get_keywords_db_path()
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]
            except Exception:
                return []
        return []

    def _save_keywords_db(self, keywords: List[str]):
        path = self._get_keywords_db_path()
        try:
            with open(path, 'w', encoding='utf-8') as f:
                for kw in sorted(set(keywords), key=lambda x: x.lower()):
                    f.write(kw + '\n')
        except Exception:
            pass

    def _normalize_keyword(self, kw: str) -> str:
        kw = kw.strip()
        # 英文关键词：统一小写，去除多余空白与尾部标点
        kw = re.sub(r'[\s\u3000]+', ' ', kw)
        kw = kw.strip().strip('.,;:')
        return kw.lower()

    def _find_similar_in_db(self, db: List[str], new_kw: str, threshold: float = 0.88) -> str:
        if not new_kw:
            return None
        candidates = difflib.get_close_matches(new_kw, [self._normalize_keyword(k) for k in db], n=1, cutoff=threshold)
        if candidates:
            # 映射回原始大小写形式（优先第一个匹配项）
            norm = candidates[0]
            for k in db:
                if self._normalize_keyword(k) == norm:
                    return k
        return None

    def _merge_keywords_with_db(self, extracted_keywords: List[str]) -> Tuple[List[str], List[str]]:
        """
        将提取的关键词与关键词库进行合并去重，返回：
        - canonical_keywords: 替换/合并后的关键词列表（用于写回 YAML）
        - updated_db: 更新后的关键词库（若有新增）
        """
        db = self._load_keywords_db()
        canonical_list: List[str] = []

        for kw in extracted_keywords:
            clean = self._normalize_keyword(kw)
            if not clean:
                continue
            similar = self._find_similar_in_db(db, clean)
            if similar:
                # 使用库中的标准词形
                if similar not in canonical_list:
                    canonical_list.append(similar)
            else:
                # 新关键词：加入库与结果
                db.append(kw)
                if kw not in canonical_list:
                    canonical_list.append(kw)

        # 保存更新的关键词库
        self._save_keywords_db(db)
        return canonical_list, db

    def _generate_yaml_header(self) -> Generator:
        """基于论文内容与已得分析，生成 YAML 头部（核心元信息）"""
        try:
            prompt = (
                "请基于以下论文内容与分析要点，提取论文核心元信息并输出 YAML Front Matter：\n\n"
                f"论文全文内容片段：\n{self.paper_content}\n\n"
                "若有可用的分析要点：\n"
            )

            # 将已有结果简要拼接，辅助提取
            for q in self.questions:
                if q.id in self.results:
                    prompt += f"- {q.description}: {self.results[q.id][:400]}\n"

            prompt += (
                "\n严格输出 YAML（不使用代码块围栏），字段如下：\n"
                "title: 原文标题（尽量英文原题,标题需要有引号包裹）\n"
                "title_zh: 中文标题（若可）\n"
                "authors: [作者英文名列表]\n"
                "affiliation_zh: 第一作者单位（中文）\n"
                "keywords: [英文关键词列表]\n"
                "urls: [论文链接, Github链接或None]\n"
                "source_code: [源码链接, None]\n"
                "read_status: [已阅读, 未阅读]\n"
                "stars: [⭐⭐⭐⭐⭐, ⭐⭐⭐⭐, ⭐⭐⭐, ⭐⭐, ⭐]\n"
                "仅输出以 --- 开始、以 --- 结束的 YAML Front Matter，不要附加其他文本。默认stars为⭐⭐⭐，read_status为未阅读。"
            )

            yaml_str = yield from request_gpt_model_in_new_thread_with_ui_alive(
                inputs=prompt,
                inputs_show_user="生成论文核心信息 YAML 头",
                llm_kwargs=self.llm_kwargs,
                chatbot=self.chatbot,
                history=[],
                sys_prompt=(
                    "你是论文信息抽取助手。请仅输出 YAML Front Matter，"
                    "键名固定且顺序不限，注意 authors/keywords/urls 应为列表。"
                )
            )

            # 简单校验，确保包含 YAML 分隔符
            if isinstance(yaml_str, str) and yaml_str.strip().startswith("---") and yaml_str.strip().endswith("---"):
                # 解析并规范化 keywords 列表
                text = yaml_str.strip()
                m = re.search(r"^keywords:\s*\[(.*?)\]\s*$", text, flags=re.MULTILINE)
                if m:
                    inner = m.group(1).strip()
                    # 简单解析列表内容，支持带引号或不带引号的英文关键词
                    # 拆分逗号，同时去掉包裹引号
                    raw_list = [x.strip().strip('"\'\'') for x in inner.split(',') if x.strip()]
                    merged, _ = self._merge_keywords_with_db(raw_list)
                    # 以原样式写回（使用引号包裹，避免 YAML 解析问题）
                    rebuilt = ', '.join([f'"{k}"' for k in merged])
                    text = re.sub(r"^keywords:\s*\[(.*?)\]\s*$", f"keywords: [{rebuilt}]", text, flags=re.MULTILINE)
                return text
            return None

        except Exception as e:
            self.chatbot.append(["警告", f"生成 YAML 头失败: {str(e)}"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return None
    def _update_category_json(self, llm_answer: str):
        """
        解析 LLM 返回的归属/新增指令，并更新 paper.json
        """
        json_path = os.path.join(os.path.dirname(__file__), 'paper.json')

        # 1) 新增一级
        m1 = re.search(r'新增一级：(.+?) *-> *\[(.+?)\]', llm_answer)
        if m1:
            new_main = m1.group(1).strip()
            new_subs = [s.strip() for s in m1.group(2).split(',')]
            self.category_tree[new_main] = new_subs
        else:
            # 2) 新增二级
            m2 = re.search(r'新增二级：(.+?) *-> *(.+)', llm_answer)
            if m2:
                main_cat = m2.group(1).strip()
                new_sub = m2.group(2).strip()
                if main_cat in self.category_tree and new_sub not in self.category_tree[main_cat]:
                    self.category_tree[main_cat].append(new_sub)

        # 写回
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.category_tree, f, ensure_ascii=False, indent=4)


    def _load_paper(self, paper_path: str) -> Generator:
        from crazy_functions.doc_fns.text_content_loader import TextContentLoader
        """加载论文内容"""
        yield from update_ui(chatbot=self.chatbot, history=self.history)

        # 保存论文文件路径
        self.paper_file_path = paper_path

        # 使用TextContentLoader读取文件
        loader = TextContentLoader(self.chatbot, self.history)

        yield from loader.execute_single_file(paper_path)

        # 获取加载的内容
        if len(self.history) >= 2 and self.history[-2]:
            self.paper_content = self.history[-2]
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return True
        else:
            self.chatbot.append(["错误", "无法读取论文内容，请检查文件是否有效"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return False

    def _analyze_question(self, question: PaperQuestion) -> Generator:
        """分析单个问题 - 直接显示问题和答案"""
        try:
            prompt = f"请基于以下论文内容回答问题：\n\n{self.paper_content}\n\n问题：{question.question}"

            response = yield from request_gpt_model_in_new_thread_with_ui_alive(
                inputs=prompt,
                inputs_show_user=question.question,
                llm_kwargs=self.llm_kwargs,
                chatbot=self.chatbot,
                history=[],
                sys_prompt="你是一个专业的科研论文分析助手，需要仔细阅读论文内容并回答问题。请保持客观、准确，并基于论文内容提供深入分析。"
            )

            if response:
                self.results[question.id] = response

                # 如果是分类归属问题，自动更新 paper.json
                if question.id == "category_assignment":
                    self._update_category_json(response)

                return True
            return False

        except Exception as e:
            self.chatbot.append(["错误", f"分析问题时出错: {str(e)}"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return False


    def _generate_summary(self) -> Generator:
        """生成最终总结报告"""
        self.chatbot.append(["生成报告", "正在整合分析结果，生成最终报告..."])
        yield from update_ui(chatbot=self.chatbot, history=self.history)

        summary_prompt = "请基于以下对论文的各个方面的分析，生成一份全面的论文解读报告。报告应该简明扼要地呈现论文的关键内容，并保持逻辑连贯性。"

        for q in self.questions:
            if q.id in self.results:
                summary_prompt += f"\n\n关于{q.description}的分析:\n{self.results[q.id]}"

        try:
            # 使用单线程版本的请求函数，可以在前端实时显示生成结果
            response = yield from request_gpt_model_in_new_thread_with_ui_alive(
                inputs=summary_prompt,
                inputs_show_user="生成论文解读报告",
                llm_kwargs=self.llm_kwargs,
                chatbot=self.chatbot,
                history=[],
                sys_prompt="你是一个科研论文解读专家，请将多个方面的分析整合为一份完整、连贯、有条理的报告。报告应当重点突出，层次分明，并且保持学术性和客观性。若分析中包含 Mermaid 代码块（```mermaid ...```），请原样保留，不要改写为其他格式。"
            )

            if response:
                return response
            return "报告生成失败"

        except Exception as e:
            self.chatbot.append(["错误", f"生成报告时出错: {str(e)}"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return "报告生成失败: " + str(e)

    def save_report(self, report: str, paper_file_path: str = None) -> str:
        """保存分析报告，返回保存的文件路径"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 获取PDF文件名（不含扩展名）
        pdf_filename = "未知论文"
        if paper_file_path and os.path.exists(paper_file_path):
            pdf_filename = os.path.splitext(os.path.basename(paper_file_path))[0]
            # 清理文件名中的特殊字符，只保留字母、数字、中文和下划线
            import re
            pdf_filename = re.sub(r'[^\w\u4e00-\u9fff]', '_', pdf_filename)
            # 如果文件名过长，截取前50个字符
            if len(pdf_filename) > 50:
                pdf_filename = pdf_filename[:50]

        # 保存为Markdown文件
        try:
            md_parts = []
            # 标题与整体报告（稍后在前加入 YAML 头）
            md_parts.append(f"论文快速解读报告\n\n{report}")

            # 优先写入：PPT 极简摘要（若有）
            if "core_idea_ppt_md" in self.results:
                md_parts.append(f"\n\n## PPT 摘要\n\n{self.results['core_idea_ppt_md']}")

            # 其次写入：核心流程图（Mermaid）（若有，保持代码块原样）
            if "core_algorithm_flowcharts" in self.results:
                md_parts.append(f"\n\n## 核心流程图\n\n{self.results['core_algorithm_flowcharts']}")

            # 其余分析项按问题列表顺序写入，但跳过已写入的两个
            for q in self.questions:
                if q.id in self.results and q.id not in {"core_idea_ppt_md", "core_algorithm_flowcharts"}:
                    md_parts.append(f"\n\n## {q.description}\n\n{self.results[q.id]}")

            md_content = "".join(md_parts)

            # 若已生成 YAML 头，则置于文首
            if hasattr(self, 'yaml_header') and self.yaml_header:
                md_content = f"{self.yaml_header}\n\n" + md_content

            result_file = write_history_to_file(
                history=[md_content],
                file_basename=f"{timestamp}_{pdf_filename}_解读报告.md"
            )

            if result_file and os.path.exists(result_file):
                promote_file_to_downloadzone(result_file, chatbot=self.chatbot)
                return result_file
            else:
                return None
        except Exception as e:
            self.chatbot.append(["警告", f"保存报告失败: {str(e)}"])
            update_ui(chatbot=self.chatbot, history=self.history)
            return None

    def analyze_paper(self, paper_path: str) -> Generator:
        """分析单篇论文主流程"""
        # 加载论文
        success = yield from self._load_paper(paper_path)
        if not success:
            return None

        # 分析关键问题 - 直接询问每个问题，不显示进度信息
        for question in self.questions:
            yield from self._analyze_question(question)

        # 生成总结报告
        final_report = yield from self._generate_summary()

        # 生成 YAML 头
        self.yaml_header = yield from self._generate_yaml_header()

        # 保存报告
        saved_file = self.save_report(final_report, self.paper_file_path)
        
        return saved_file


def _find_paper_files(path: str) -> List[str]:
    """查找路径中的所有论文文件"""
    paper_files = []
    
    if os.path.isfile(path):
        # 如果是单个文件，检查是否为支持的格式
        file_ext = os.path.splitext(path)[1].lower()
        if file_ext in ['.pdf', '.docx', '.doc', '.txt', '.md', '.tex']:
            paper_files.append(path)
        return paper_files

    # 如果是目录，递归搜索所有支持的论文文件
    if os.path.isdir(path):
        supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md', '.tex']
        
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in supported_extensions:
                    paper_files.append(file_path)
    
    return paper_files


def download_paper_by_id(paper_info, chatbot, history) -> str:
    """下载论文并返回保存路径"""
    from crazy_functions.review_fns.data_sources.scihub_source import SciHub
    id_type, paper_id = paper_info

    # 创建保存目录 - 使用时间戳创建唯一文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    user_name = chatbot.get_user() if hasattr(chatbot, 'get_user') else "default"
    from toolbox import get_log_folder, get_user
    base_save_dir = get_log_folder(get_user(chatbot), plugin_name='paper_download')
    save_dir = os.path.join(base_save_dir, f"papers_{timestamp}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = Path(save_dir)

    chatbot.append([f"下载论文", f"正在下载{'arXiv' if id_type == 'arxiv' else 'DOI'} {paper_id} 的论文..."])
    update_ui(chatbot=chatbot, history=history)

    pdf_path = None

    try:
        if id_type == 'arxiv':
            # 使用改进的arxiv查询方法
            formatted_id = format_arxiv_id(paper_id)
            paper_result = get_arxiv_paper(formatted_id)

            if not paper_result:
                chatbot.append([f"下载失败", f"未找到arXiv论文: {paper_id}"])
                update_ui(chatbot=chatbot, history=history)
                return None

            # 下载PDF
            filename = f"arxiv_{paper_id.replace('/', '_')}.pdf"
            pdf_path = str(save_path / filename)
            paper_result.download_pdf(filename=pdf_path)

        else:  # doi
            # 下载DOI
            sci_hub = SciHub(
                doi=paper_id,
                path=save_path
            )
            pdf_path = sci_hub.fetch()

        # 检查下载结果
        if pdf_path and os.path.exists(pdf_path):
            promote_file_to_downloadzone(pdf_path, chatbot=chatbot)
            chatbot.append([f"下载成功", f"已成功下载论文: {os.path.basename(pdf_path)}"])
            update_ui(chatbot=chatbot, history=history)
            return pdf_path
        else:
            chatbot.append([f"下载失败", f"论文下载失败: {paper_id}"])
            update_ui(chatbot=chatbot, history=history)
            return None

    except Exception as e:
        chatbot.append([f"下载错误", f"下载论文时出错: {str(e)}"])
        update_ui(chatbot=chatbot, history=history)
        return None


@CatchException
def 批量论文速读(txt: str, llm_kwargs: Dict, plugin_kwargs: Dict, chatbot: List,
             history: List, system_prompt: str, user_request: str):
    """主函数 - 批量论文速读"""
    # 初始化分析器
    chatbot.append(["函数插件功能及使用方式", "批量论文速读：批量分析多个论文文件，为每篇论文生成独立的速读报告，适用于大量论文的快速理解。 <br><br>📋 使用方式：<br>1、输入包含多个PDF文件的文件夹路径<br>2、或者输入多个论文ID（DOI或arXiv ID），用逗号分隔<br>3、点击插件开始批量分析"])
    yield from update_ui(chatbot=chatbot, history=history)

    paper_files = []

    # 检查输入是否包含论文ID（多个ID用逗号分隔）
    if ',' in txt:
        # 处理多个论文ID
        paper_ids = [id.strip() for id in txt.split(',') if id.strip()]
        chatbot.append(["检测到多个论文ID", f"检测到 {len(paper_ids)} 个论文ID，准备批量下载..."])
        yield from update_ui(chatbot=chatbot, history=history)

        for i, paper_id in enumerate(paper_ids):
            paper_info = extract_paper_id(paper_id)
            if paper_info:
                chatbot.append([f"下载论文 {i+1}/{len(paper_ids)}", f"正在下载 {'arXiv' if paper_info[0] == 'arxiv' else 'DOI'} ID: {paper_info[1]}..."])
                yield from update_ui(chatbot=chatbot, history=history)

                paper_file = download_paper_by_id(paper_info, chatbot, history)
                if paper_file:
                    paper_files.append(paper_file)
                else:
                    chatbot.append([f"下载失败", f"无法下载论文: {paper_id}"])
                    yield from update_ui(chatbot=chatbot, history=history)
            else:
                chatbot.append([f"ID格式错误", f"无法识别论文ID格式: {paper_id}"])
                yield from update_ui(chatbot=chatbot, history=history)
    else:
        # 检查单个论文ID
        paper_info = extract_paper_id(txt)
        if paper_info:
            # 单个论文ID
            chatbot.append(["检测到论文ID", f"检测到{'arXiv' if paper_info[0] == 'arxiv' else 'DOI'} ID: {paper_info[1]}，准备下载论文..."])
            yield from update_ui(chatbot=chatbot, history=history)

            paper_file = download_paper_by_id(paper_info, chatbot, history)
            if paper_file:
                paper_files.append(paper_file)
            else:
                report_exception(chatbot, history, a=f"下载论文失败", b=f"无法下载{'arXiv' if paper_info[0] == 'arxiv' else 'DOI'}论文: {paper_info[1]}")
                yield from update_ui(chatbot=chatbot, history=history)
                return
        else:
            # 检查输入路径
            if not os.path.exists(txt):
                report_exception(chatbot, history, a=f"批量解析论文: {txt}", b=f"找不到文件或无权访问: {txt}")
                yield from update_ui(chatbot=chatbot, history=history)
                return

            # 验证路径安全性
            user_name = chatbot.get_user()
            validate_path_safety(txt, user_name)

            # 查找所有论文文件
            paper_files = _find_paper_files(txt)

            if not paper_files:
                report_exception(chatbot, history, a=f"批量解析论文", b=f"在路径 {txt} 中未找到支持的论文文件")
                yield from update_ui(chatbot=chatbot, history=history)
                return

    yield from update_ui(chatbot=chatbot, history=history)

    # 开始批量分析
    if not paper_files:
        chatbot.append(["错误", "没有找到任何可分析的论文文件"])
        yield from update_ui(chatbot=chatbot, history=history)
        return

    chatbot.append(["开始批量分析", f"找到 {len(paper_files)} 篇论文，开始批量分析..."])
    yield from update_ui(chatbot=chatbot, history=history)

    # 创建批量分析器
    analyzer = BatchPaperAnalyzer(llm_kwargs, plugin_kwargs, chatbot, history, system_prompt)
    
    # 批量分析每篇论文
    successful_reports = []
    failed_papers = []
    
    for i, paper_file in enumerate(paper_files):
        try:
            chatbot.append([f"分析论文 {i+1}/{len(paper_files)}", f"正在分析: {os.path.basename(paper_file)}"])
            yield from update_ui(chatbot=chatbot, history=history)
            
            # 分析单篇论文
            saved_file = yield from analyzer.analyze_paper(paper_file)
            
            if saved_file:
                successful_reports.append((os.path.basename(paper_file), saved_file))
                chatbot.append([f"完成论文 {i+1}/{len(paper_files)}", f"成功分析并保存报告: {os.path.basename(saved_file)}"])
            else:
                failed_papers.append(os.path.basename(paper_file))
                chatbot.append([f"失败论文 {i+1}/{len(paper_files)}", f"分析失败: {os.path.basename(paper_file)}"])
            
            yield from update_ui(chatbot=chatbot, history=history)
            
        except Exception as e:
            failed_papers.append(os.path.basename(paper_file))
            chatbot.append([f"错误论文 {i+1}/{len(paper_files)}", f"分析出错: {os.path.basename(paper_file)} - {str(e)}"])
            yield from update_ui(chatbot=chatbot, history=history)

    # 生成批量分析总结
    summary = f"批量分析完成！\n\n"
    summary += f"📊 分析统计：\n"
    summary += f"- 总论文数：{len(paper_files)}\n"
    summary += f"- 成功分析：{len(successful_reports)}\n"
    summary += f"- 分析失败：{len(failed_papers)}\n\n"
    
    if successful_reports:
        summary += f"✅ 成功生成报告：\n"
        for paper_name, report_path in successful_reports:
            summary += f"- {paper_name} → {os.path.basename(report_path)}\n"
    
    if failed_papers:
        summary += f"\n❌ 分析失败的论文：\n"
        for paper_name in failed_papers:
            summary += f"- {paper_name}\n"

    chatbot.append(["批量分析完成", summary])
    yield from update_ui(chatbot=chatbot, history=history) 