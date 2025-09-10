import os
import re
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Generator

from crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive
from toolbox import update_ui, promote_file_to_downloadzone, write_history_to_file, CatchException, report_exception
from shared_utils.fastapi_server import validate_path_safety
from crazy_functions.paper_fns.paper_download import extract_paper_id, get_arxiv_paper, format_arxiv_id


@dataclass
class DeepReadQuestion:
    """论文精读问题项"""
    id: str
    question: str
    importance: int
    description: str


class BatchPaperDetailAnalyzer:
    """批量论文精读分析器"""

    def __init__(self, llm_kwargs: Dict, plugin_kwargs: Dict, chatbot: List, history: List, system_prompt: str):
        self.llm_kwargs = llm_kwargs
        self.plugin_kwargs = plugin_kwargs
        self.chatbot = chatbot
        self.history = history
        self.system_prompt = system_prompt
        self.paper_content = ""
        self.results: Dict[str, str] = {}
        self.paper_file_path: str = None
        self.secondary_category: str = None

        # 精读维度（更深入的技术细节、可复现性、理论依据等）
        self.questions: List[DeepReadQuestion] = [
            DeepReadQuestion(
                id="problem_statement_and_contributions",
                description="问题定义与贡献",
                importance=5,
                question=(
                    "请严谨概述论文要解决的问题、形式化定义与边界条件；"
                    "梳理论文的核心贡献（按重要性排序），并逐条说明其与已有工作的本质区别。"
                ),
            ),
            DeepReadQuestion(
                id="method_derivation_and_theory",
                description="方法推导与理论保障",
                importance=5,
                question=(
                    "请对核心方法进行公式级精读：给出关键符号定义、损失函数/目标函数，"
                    "推导主命题（或定理/引理）的关键步骤与必要假设；若给出收敛性/复杂度/一致性结论，"
                    "请明确其适用前提与局限。必要时给出简化版等价表述方便实现。"
                ),
            ),
            DeepReadQuestion(
                id="assumptions_and_threats",
                description="关键假设与有效性威胁",
                importance=4,
                question=(
                    "列出论文显式或隐含假设（数据分布、独立性、可获取性、硬件条件等），"
                    "分析这些假设与现实差距可能带来的失效情形；"
                    "补充作者未充分讨论但可能重要的有效性威胁。"
                ),
            ),
            DeepReadQuestion(
                id="experiments_reproduction_plan",
                description="实验设计与可重复性要点（不含代码）",
                importance=5,
                question=(
                    "请总结实验设计与可重复性关键要点（不涉及任何代码/命令）：数据集版本与获取方式、预处理流程、"
                    "核心超参数名称（无需给出具体数值）、训练与评测流程概述、对比方法与消融实验设计、"
                    "资源需求量级（如GPU/CPU/时长）。"
                ),
            ),
            DeepReadQuestion(
                id="dataset_and_license",
                description="数据集与许可",
                importance=3,
                question=(
                    "请罗列论文用到的数据/模型资源及其许可证；说明是否存在闭源依赖或不可获得资源，"
                    "并提供相应替代方案建议。"
                ),
            ),
            DeepReadQuestion(
                id="results_and_ablations",
                description="结果复核与消融洞察",
                importance=4,
                question=(
                    "请对主要结果进行要点复核：是否满足统计显著性、是否有方差报告；"
                    "从消融实验中提炼最关键的影响因素与交互效应，指出可能的误导性结论。"
                ),
            ),
            DeepReadQuestion(
                id="limitations_future_and_impact",
                description="限制、未来方向与影响",
                importance=3,
                question=(
                    "总结论文的主要局限、潜在风险或伦理问题；提出可操作的改进方向与后续研究假设；"
                    "评估其对学术与产业的中短期影响路径。"
                ),
            ),
            DeepReadQuestion(
                id="mermaid_flowcharts",
                description="核心流程图（Mermaid）",
                importance=4,
                question=(
                    "请给出与实现强相关的流程图（可多个模块）\n"
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
                    "flowchart TD\n"
                    "    A[\"输入\"] --> B(\"处理\")\n"
                    "    B --> C{\"是否满足条件\"}\n"
                    "    C --> D[\"输出1\"]\n"
                    "    C --> |\"否\"| E[\"输出2\"]\n"
                    "```"
                ),
            ),
            DeepReadQuestion(
                id="exec_summary_md",
                description="精读要点摘要（Markdown）",
                importance=5,
                question=(
                    "给出极简 Markdown 摘要（不包含任何代码/命令）：\n"
                    "- 一句话总述方法与作用\n"
                    "- 三到五条方法要点（输入/步骤/输出）\n"
                    "- 三到五条复现要点（数据/超参名称/资源量级/时长）"
                ),
            ),
        ]

        self.questions.sort(key=lambda q: q.importance, reverse=True)

    # ---------- 关键词库工具（与速读版一致） ----------
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
        kw = re.sub(r'[\s\u3000]+', ' ', kw)
        kw = kw.strip().strip('.,;:')
        return kw.lower()

    def _find_similar_in_db(self, db: List[str], new_kw: str, threshold: float = 0.88) -> str:
        import difflib
        if not new_kw:
            return None
        candidates = difflib.get_close_matches(new_kw, [self._normalize_keyword(k) for k in db], n=1, cutoff=threshold)
        if candidates:
            norm = candidates[0]
            for k in db:
                if self._normalize_keyword(k) == norm:
                    return k
        return None

    def _merge_keywords_with_db(self, extracted_keywords: List[str]):
        db = self._load_keywords_db()
        canonical_list: List[str] = []
        for kw in extracted_keywords:
            clean = self._normalize_keyword(kw)
            if not clean:
                continue
            similar = self._find_similar_in_db(db, clean)
            if similar:
                if similar not in canonical_list:
                    canonical_list.append(similar)
            else:
                db.append(kw)
                if kw not in canonical_list:
                    canonical_list.append(kw)
        self._save_keywords_db(db)
        return canonical_list, db

    def _generate_yaml_header(self) -> Generator:
        """基于论文内容与已得分析，生成 YAML Front Matter"""
        try:
            prompt = (
                "请基于以下论文内容与分析要点，提取论文核心元信息并输出 YAML Front Matter：\n\n"
                f"论文全文内容片段：\n{self.paper_content}\n\n"
                "若有可用的分析要点：\n"
            )
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
                "doi: [DOI链接, None]\n"
                "journal_or_conference: [期刊或会议名称, None]\n"
                "year: [年份, None]\n"
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

            if isinstance(yaml_str, str) and yaml_str.strip().startswith("---") and yaml_str.strip().endswith("---"):
                text = yaml_str.strip()
                m = re.search(r"^keywords:\s*\[(.*?)\]\s*$", text, flags=re.MULTILINE)
                if m:
                    inner = m.group(1).strip()
                    raw_list = [x.strip().strip('\"\'\'') for x in inner.split(',') if x.strip()]
                    merged, _ = self._merge_keywords_with_db(raw_list)
                    rebuilt = ', '.join([f'"{k}"' for k in merged])
                    text = re.sub(r"^keywords:\s*\[(.*?)\]\s*$", f"keywords: [{rebuilt}]", text, flags=re.MULTILINE)
                # 将本次精读使用的分类化提示（prompts）归档到 YAML 头
                try:
                    prompts_lines = ["deep_read_prompts:"]
                    for q in self.questions:
                        desc = q.description.replace('"', '\\"')
                        prompts_lines.append(f"  - id: {q.id}")
                        prompts_lines.append(f"    description: \"{desc}\"")
                        prompts_lines.append(f"    importance: {q.importance}")
                    prompts_block = "\n".join(prompts_lines) + "\n"
                    if text.endswith("---"):
                        text = text[:-3].rstrip() + "\n" + prompts_block + "---"
                except Exception:
                    pass
                # 注入“归属”二级分类到 YAML 头（仅写入分类路径本身，并用引号包裹）
                try:
                    if getattr(self, 'secondary_category', None):
                        escaped = self.secondary_category.replace('"', '\\"')
                        if text.endswith("---"):
                            text = text[:-3].rstrip() + f"\nsecondary_category: \"{escaped}\"\n---"
                except Exception:
                    pass
                # 基于 stars 推断“论文重要程度”并写入（中文等级，值用引号包裹）
                try:
                    m_star = re.search(r"^stars:\s*\[(.*?)\]\s*$", text, flags=re.MULTILINE)
                    level = None
                    if m_star:
                        inner = m_star.group(1)
                        m_seq = re.search(r"(⭐{1,5})", inner)
                        if m_seq:
                            count = len(m_seq.group(1))
                            if count >= 5:
                                level = "强烈推荐"
                            elif count == 4:
                                level = "推荐"
                            elif count == 3:
                                level = "一般"
                            elif count == 2:
                                level = "谨慎"
                            elif count == 1:
                                level = "不推荐"
                    if not level:
                        level = "一般"
                    if text.endswith("---"):
                        text = text[:-3].rstrip() + f"\n论文重要程度: \"{level}\"\n---"
                except Exception:
                    pass
                return text
            return None
        except Exception as e:
            self.chatbot.append(["警告", f"生成 YAML 头失败: {str(e)}"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return None

    def _load_paper(self, paper_path: str) -> Generator:
        from crazy_functions.doc_fns.text_content_loader import TextContentLoader
        yield from update_ui(chatbot=self.chatbot, history=self.history)
        self.paper_file_path = paper_path
        loader = TextContentLoader(self.chatbot, self.history)
        yield from loader.execute_single_file(paper_path)
        if len(self.history) >= 2 and self.history[-2]:
            self.paper_content = self.history[-2]
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return True
        self.chatbot.append(["错误", "无法读取论文内容，请检查文件是否有效"])
        yield from update_ui(chatbot=self.chatbot, history=self.history)
        return False

    def _ask(self, q: DeepReadQuestion) -> Generator:
        try:
            prompt = (
                "请基于以下论文内容进行精读分析，并严格围绕问题作答。\n"
                "注意：请避免提供任何代码、伪代码、命令行或具体实现细节；"
                "若输出流程图，须使用 ```mermaid 代码块，其余回答保持自然语言。\n\n"
                
                f"论文内容：\n{self.paper_content}\n\n"
                f"问题：{q.question}"
            )
            resp = yield from request_gpt_model_in_new_thread_with_ui_alive(
                inputs=prompt,
                inputs_show_user=q.question,
                llm_kwargs=self.llm_kwargs,
                chatbot=self.chatbot,
                history=[],
                sys_prompt=(
                    "你是资深研究员，输出以概念与方法论层面为主，不包含任何代码或伪代码。"
                    "如涉及Mermaid流程图，请使用```mermaid 包裹并保持语法正确，其余保持自然语言。"
                ),
            )
            if resp:
                self.results[q.id] = resp
                return True
            return False
        except Exception as e:
            self.chatbot.append(["错误", f"精读问题分析失败: {str(e)}"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return False

    def _generate_report(self) -> Generator:
        self.chatbot.append(["生成报告", "正在整合精读结果，生成深度技术报告..."])
        yield from update_ui(chatbot=self.chatbot, history=self.history)

        prompt = (
            "请将以下精读分析整理为完整的技术报告，层次清晰，突出核心思想与实验设计要点，"
            "不包含任何代码/伪代码/命令行。若包含```mermaid 代码块，请原样保留。"
        )
        for q in self.questions:
            if q.id in self.results:
                prompt += f"\n\n[{q.description}]\n{self.results[q.id]}"

        resp = yield from request_gpt_model_in_new_thread_with_ui_alive(
            inputs=prompt,
            inputs_show_user="生成论文精读技术报告",
            llm_kwargs=self.llm_kwargs,
            chatbot=self.chatbot,
            history=[],
            sys_prompt=(
                "以工程复现为目标组织报告：背景极简，方法与实现细节充分，"
                "条理分明，包含必要的清单与步骤。"
            ),
        )
        return resp or "报告生成失败"

    def _extract_secondary_category(self, report: str) -> str:
        """从报告中提取“归属：”后的二级分类文本，只保留类似
        “7. 机器学习辅助设计 (ML-Aided RF Design) -> 系统级建模与快速综合”。
        """
        try:
            if not isinstance(report, str):
                return None
            m = re.search(r"^归属：\s*([^\r\n]+)", report, flags=re.MULTILINE)
            if not m:
                return None
            category_line = m.group(1).strip()
            category_line = re.sub(r"[\s\u3000]+$", "", category_line)
            return category_line if category_line else None
        except Exception:
            return None

    def save_report(self, report: str) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        pdf_basename = "未知论文"
        if self.paper_file_path and os.path.exists(self.paper_file_path):
            pdf_basename = os.path.splitext(os.path.basename(self.paper_file_path))[0]
            pdf_basename = re.sub(r'[^\w\u4e00-\u9fff]', '_', pdf_basename)
            if len(pdf_basename) > 50:
                pdf_basename = pdf_basename[:50]

        parts: List[str] = []
        parts.append(f"论文精读技术报告\n\n{report}")
        # 优先追加执行级摘要与流程图
        if "exec_summary_md" in self.results:
            parts.append(f"\n\n## 执行级摘要\n\n{self.results['exec_summary_md']}")
        if "mermaid_flowcharts" in self.results:
            parts.append(f"\n\n## 核心流程图\n\n{self.results['mermaid_flowcharts']}")
        # 追加其余维度
        for q in self.questions:
            if q.id in self.results and q.id not in {"exec_summary_md", "mermaid_flowcharts"}:
                parts.append(f"\n\n## {q.description}\n\n{self.results[q.id]}")

        content = "".join(parts)
        if hasattr(self, 'yaml_header') and self.yaml_header:
            content = f"{self.yaml_header}\n\n" + content
        result_file = write_history_to_file(
            history=[content],
            file_basename=f"{timestamp}_{pdf_basename}_精读报告.md",
        )
        if result_file and os.path.exists(result_file):
            promote_file_to_downloadzone(result_file, chatbot=self.chatbot)
            return result_file
        return None

    def analyze_paper(self, paper_path: str) -> Generator:
        ok = yield from self._load_paper(paper_path)
        if not ok:
            return None
        for q in self.questions:
            yield from self._ask(q)
        report = yield from self._generate_report()
        # 从报告中提取二级分类归属
        self.secondary_category = self._extract_secondary_category(report)
        # 生成 YAML 头
        self.yaml_header = yield from self._generate_yaml_header()
        saved = self.save_report(report)
        return saved


def _find_paper_files(path: str) -> List[str]:
    files: List[str] = []
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in [".pdf", ".docx", ".doc", ".txt", ".md", ".tex"]:
            files.append(path)
        return files
    if os.path.isdir(path):
        exts = [".pdf", ".docx", ".doc", ".txt", ".md", ".tex"]
        for root, _dirs, fnames in os.walk(path):
            for fname in fnames:
                fpath = os.path.join(root, fname)
                if os.path.splitext(fname)[1].lower() in exts:
                    files.append(fpath)
    return files


def _download_paper_by_id(paper_info, chatbot, history) -> str:
    from crazy_functions.review_fns.data_sources.scihub_source import SciHub
    id_type, paper_id = paper_info

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    user_name = chatbot.get_user() if hasattr(chatbot, 'get_user') else "default"
    from toolbox import get_log_folder, get_user
    base_save_dir = get_log_folder(get_user(chatbot), plugin_name='paper_download')
    save_dir = os.path.join(base_save_dir, f"papers_{timestamp}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = Path(save_dir)

    chatbot.append(["下载论文", f"正在下载{'arXiv' if id_type == 'arxiv' else 'DOI'} {paper_id} 的论文..."])
    update_ui(chatbot=chatbot, history=history)

    pdf_path = None
    try:
        if id_type == 'arxiv':
            formatted_id = format_arxiv_id(paper_id)
            paper_result = get_arxiv_paper(formatted_id)
            if not paper_result:
                chatbot.append(["下载失败", f"未找到arXiv论文: {paper_id}"])
                update_ui(chatbot=chatbot, history=history)
                return None
            filename = f"arxiv_{paper_id.replace('/', '_')}.pdf"
            pdf_path = str(save_path / filename)
            paper_result.download_pdf(filename=pdf_path)
        else:
            sci_hub = SciHub(doi=paper_id, path=save_path)
            pdf_path = sci_hub.fetch()

        if pdf_path and os.path.exists(pdf_path):
            promote_file_to_downloadzone(pdf_path, chatbot=chatbot)
            chatbot.append(["下载成功", f"已成功下载论文: {os.path.basename(pdf_path)}"])
            update_ui(chatbot=chatbot, history=history)
            return pdf_path
        chatbot.append(["下载失败", f"论文下载失败: {paper_id}"])
        update_ui(chatbot=chatbot, history=history)
        return None
    except Exception as e:
        chatbot.append(["下载错误", f"下载论文时出错: {str(e)}"])
        update_ui(chatbot=chatbot, history=history)
        return None


@CatchException
def 批量论文精读(txt: str, llm_kwargs: Dict, plugin_kwargs: Dict, chatbot: List,
            history: List, system_prompt: str, user_request: str):
    """主函数 - 批量论文精读"""
    chatbot.append([
        "函数插件功能及使用方式",
        (
            "批量论文精读：对多个论文文件进行深入阅读与技术复盘，输出面向实现与复现的深度报告。\n\n"
            "使用方式：\n1) 输入包含多个PDF的文件夹路径；\n2) 或输入多个论文ID（DOI或arXiv），用逗号分隔；\n3) 点击开始。\n\n"
            "注意事项：\n- 若需要输出公式，请使用 LaTeX 数学格式：行内公式用 $...$，行间公式用 $$...$$。"
        ),
    ])
    yield from update_ui(chatbot=chatbot, history=history)

    paper_files: List[str] = []

    if ',' in txt:
        paper_ids = [pid.strip() for pid in txt.split(',') if pid.strip()]
        chatbot.append(["检测到多个论文ID", f"检测到 {len(paper_ids)} 个论文ID，准备批量下载..."])
        yield from update_ui(chatbot=chatbot, history=history)
        for i, pid in enumerate(paper_ids):
            paper_info = extract_paper_id(pid)
            if paper_info:
                chatbot.append([f"下载论文 {i+1}/{len(paper_ids)}", f"正在下载 {'arXiv' if paper_info[0] == 'arxiv' else 'DOI'} ID: {paper_info[1]}..."])
                yield from update_ui(chatbot=chatbot, history=history)
                p = _download_paper_by_id(paper_info, chatbot, history)
                if p:
                    paper_files.append(p)
                else:
                    chatbot.append(["下载失败", f"无法下载论文: {pid}"])
                    yield from update_ui(chatbot=chatbot, history=history)
            else:
                chatbot.append(["ID格式错误", f"无法识别论文ID格式: {pid}"])
                yield from update_ui(chatbot=chatbot, history=history)
    else:
        paper_info = extract_paper_id(txt)
        if paper_info:
            chatbot.append(["检测到论文ID", f"检测到{'arXiv' if paper_info[0] == 'arxiv' else 'DOI'} ID: {paper_info[1]}，准备下载论文..."])
            yield from update_ui(chatbot=chatbot, history=history)
            p = _download_paper_by_id(paper_info, chatbot, history)
            if p:
                paper_files.append(p)
            else:
                report_exception(chatbot, history, a="下载论文失败", b=f"无法下载{'arXiv' if paper_info[0] == 'arxiv' else 'DOI'}论文: {paper_info[1]}")
                yield from update_ui(chatbot=chatbot, history=history)
                return
        else:
            if not os.path.exists(txt):
                report_exception(chatbot, history, a=f"批量精读论文: {txt}", b=f"找不到文件或无权访问: {txt}")
                yield from update_ui(chatbot=chatbot, history=history)
                return
            user_name = chatbot.get_user()
            validate_path_safety(txt, user_name)
            paper_files = _find_paper_files(txt)
            if not paper_files:
                report_exception(chatbot, history, a="批量精读论文", b=f"在路径 {txt} 中未找到支持的论文文件")
                yield from update_ui(chatbot=chatbot, history=history)
                return

    yield from update_ui(chatbot=chatbot, history=history)

    if not paper_files:
        chatbot.append(["错误", "没有找到任何可分析的论文文件"])
        yield from update_ui(chatbot=chatbot, history=history)
        return

    chatbot.append(["开始批量精读", f"找到 {len(paper_files)} 篇论文，开始深入分析..."])
    yield from update_ui(chatbot=chatbot, history=history)

    analyzer = BatchPaperDetailAnalyzer(llm_kwargs, plugin_kwargs, chatbot, history, system_prompt)

    successes: List[str] = []
    failures: List[str] = []

    for i, paper_file in enumerate(paper_files):
        try:
            chatbot.append([f"精读论文 {i+1}/{len(paper_files)}", f"正在精读: {os.path.basename(paper_file)}"])
            yield from update_ui(chatbot=chatbot, history=history)
            outfile = yield from analyzer.analyze_paper(paper_file)
            if outfile:
                successes.append(outfile)
                chatbot.append([f"完成论文 {i+1}/{len(paper_files)}", f"成功生成报告: {os.path.basename(outfile)}"])
            else:
                failures.append(os.path.basename(paper_file))
                chatbot.append([f"失败论文 {i+1}/{len(paper_files)}", f"分析失败: {os.path.basename(paper_file)}"])
            yield from update_ui(chatbot=chatbot, history=history)
        except Exception as e:
            failures.append(os.path.basename(paper_file))
            chatbot.append([f"错误论文 {i+1}/{len(paper_files)}", f"分析出错: {os.path.basename(paper_file)} - {str(e)}"])
            yield from update_ui(chatbot=chatbot, history=history)

    summary = "批量精读完成！\n\n"
    summary += "📊 分析统计：\n"
    summary += f"- 总论文数：{len(paper_files)}\n"
    summary += f"- 成功分析：{len(successes)}\n"
    summary += f"- 分析失败：{len(failures)}\n\n"
    if successes:
        summary += "✅ 成功生成报告：\n"
        for p in successes:
            summary += f"- {os.path.basename(p)}\n"
    if failures:
        summary += "\n❌ 分析失败的论文：\n"
        for name in failures:
            summary += f"- {name}\n"

    chatbot.append(["批量精读完成", summary])
    yield from update_ui(chatbot=chatbot, history=history)


