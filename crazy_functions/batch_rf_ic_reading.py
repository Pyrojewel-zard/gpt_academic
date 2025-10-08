import os
import time
import glob
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Generator, Tuple
from crazy_functions.Batch_Paper_Reading import estimate_token_usage
from crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive
from toolbox import update_ui, promote_file_to_downloadzone, write_history_to_file, CatchException, report_exception
from shared_utils.fastapi_server import validate_path_safety
from crazy_functions.paper_fns.paper_download import extract_paper_id, extract_paper_ids, get_arxiv_paper, format_arxiv_id


@dataclass
class RFICQuestion:
    """射频集成电路论文分析问题类"""
    id: str  # 问题ID
    question: str  # 问题内容
    importance: int  # 重要性 (1-5，5最高)
    description: str  # 问题描述


class BatchRFICAnalyzer:
    """批量射频集成电路论文快速分析器"""

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
        self.yaml_header = None
        self.context_history: List[str] = []  # 与LLM共享的上下文（每篇论文注入一次全文）
        # 统计用：记录每次LLM交互的输入与输出
        self._token_inputs: List[str] = []
        self._token_outputs: List[str] = []

        # 定义射频集成电路论文分析问题库（专门针对RF IC领域）
        self.questions = [
            RFICQuestion(
                id="circuit_architecture",
                question="这篇RF IC论文的电路架构和拓扑结构是什么？请分析：1)核心电路架构（如LNA、PA、混频器、VCO、PLL等）；2)电路拓扑结构的特点和优势；3)关键电路模块的设计思路；4)整体系统级联和接口设计。",
                importance=5,
                description="电路架构与拓扑"
            ),
            RFICQuestion(
                id="performance_metrics",
                question="论文中RF IC的关键性能指标是什么？请详细分析：1)频率范围、带宽、增益等基本参数；2)噪声系数、线性度、效率等关键指标；3)功耗、面积、成本等设计约束；4)与现有技术的性能对比。",
                importance=5,
                description="性能指标分析"
            ),
            RFICQuestion(
                id="design_techniques",
                question="论文采用了哪些先进的RF IC设计技术？请分析：1)工艺技术选择（CMOS、SiGe、GaAs等）；2)电路设计技巧（如噪声消除、线性化技术、效率提升等）；3)版图设计和寄生效应处理；4)测试和校准方法。",
                importance=4,
                description="设计技术与工艺"
            ),
            RFICQuestion(
                id="applications_and_markets",
                question="该RF IC的应用场景和市场定位是什么？请分析：1)目标应用领域（如5G、WiFi、蓝牙、卫星通信等）；2)市场定位和竞争优势；3)技术成熟度和产业化前景；4)与现有解决方案的差异化。",
                importance=4,
                description="应用场景与市场"
            ),
            RFICQuestion(
                id="challenges_and_innovations",
                question="论文解决了哪些RF IC设计挑战？请分析：1)主要技术难点和挑战；2)创新性解决方案；3)关键技术突破；4)对行业发展的推动作用。",
                importance=3,
                description="技术挑战与创新"
            ),
            RFICQuestion(
                id="future_directions",
                question="论文对未来RF IC发展的启示是什么？请分析：1)技术发展趋势预测；2)潜在改进方向；3)与其他技术的融合机会；4)对下一代RF IC设计的指导意义。",
                importance=2,
                description="发展趋势与启示"
            ),
            RFICQuestion(
                id="worth_reading_judgment",
                question="请综合评估这篇论文是否值得精读，并从多个角度给出判断依据：1) **创新性与重要性**：论文的研究是否具有开创性？是否解决了领域内的关键问题？2) **方法可靠性**：研究方法是否严谨、可靠？实验设计是否合理？3) **论述清晰度**：论文的写作风格、图表质量和逻辑结构是否清晰易懂？4) **潜在影响**：研究成果是否可能对学术界或工业界产生较大影响？5) **综合建议**：结合以上几点，给出“强烈推荐”、“推荐”、“一般”或“不推荐”的最终评级，并简要说明理由。",
                importance=5,
                description="是否值得精读"
            ),
        ]

        # 按重要性排序
        self.questions.sort(key=lambda q: q.importance, reverse=True)

        # ---------- 关键词库工具（与 Batch_Paper_Reading 保持一致） ----------
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
        import re as _re
        kw = _re.sub(r'[\s\u3000]+', ' ', kw)
        kw = kw.strip().strip('.,;:')
        return kw.lower()

    def _find_similar_in_db(self, db: List[str], new_kw: str, threshold: float = 0.88) -> str:
        import difflib as _difflib
        if not new_kw:
            return None
        candidates = _difflib.get_close_matches(new_kw, [self._normalize_keyword(k) for k in db], n=1, cutoff=threshold)
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

    def _clean_yaml_list(self, yaml_text: str, list_fields: List[str]) -> str:
        """清理YAML文本中列表字段的None值"""
        import re
        for field in list_fields:
            # 匹配列表字段的模式
            pattern = rf"^{field}:\s*\[(.*?)\]\s*$"
            match = re.search(pattern, yaml_text, flags=re.MULTILINE)
            if match:
                inner_content = match.group(1).strip()
                if inner_content:
                    # 解析列表内容，过滤掉None值
                    items = [item.strip().strip('"\'') for item in inner_content.split(',')]
                    # 过滤掉None、空字符串和"None"
                    filtered_items = [item for item in items if item and item.lower() != 'none']
                    if filtered_items:
                        # 重新构建列表，保持引号格式
                        rebuilt = ', '.join([f'"{item}"' for item in filtered_items])
                        yaml_text = re.sub(pattern, f"{field}: [{rebuilt}]", yaml_text, flags=re.MULTILINE)
                    else:
                        # 如果列表为空，移除该字段
                        yaml_text = re.sub(rf"^{field}:\s*\[.*?\]\s*$\n?", "", yaml_text, flags=re.MULTILINE)
        return yaml_text

    def _generate_yaml_header(self) -> Generator:
        """基于论文内容与已得分析，生成 YAML 头部（核心元信息）——与 Batch_Paper_Reading 保持一致"""
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
                "doi: [DOI链接, None]\n"
                "journal_or_conference: [期刊或会议名称, None]\n"
                "year: [年份, None]\n"
                "source_code: [源码链接, None]\n"
                "read_status: [未阅读]\n"
                "stars: [⭐⭐⭐⭐⭐, ⭐⭐⭐⭐, ⭐⭐⭐, ⭐⭐, ⭐]\n"
                "仅输出以 --- 开始、以 --- 结束的 YAML Front Matter，不要附加其他文本。默认stars为⭐⭐⭐"
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
                import re as _re2
                # 解析并规范化 keywords 列表
                text = yaml_str.strip()
                m = _re2.search(r"^keywords:\s*\[(.*?)\]\s*$", text, flags=_re2.MULTILINE)
                if m:
                    inner = m.group(1).strip()
                    # 简单解析列表内容，支持带引号或不带引号的英文关键词
                    # 拆分逗号，同时去掉包裹引号
                    raw_list = [x.strip().strip('\"\'\'') for x in inner.split(',') if x.strip()]
                    merged, _ = self._merge_keywords_with_db(raw_list)
                    # 以原样式写回（使用引号包裹，避免 YAML 解析问题）
                    rebuilt = ', '.join([f'\"{k}\"' for k in merged])
                    text = _re2.sub(r"^keywords:\s*\[(.*?)\]\s*$", f"keywords: [{rebuilt}]", text, flags=_re2.MULTILINE)

                # 基于 worth_reading_judgment 提取中文"论文重要程度"和"是否精读"，若缺失再回退到默认
                try:
                    level = None
                    reading_recommendation = None
                    try:
                        judge = self.results.get("worth_reading_judgment", "")
                        if isinstance(judge, str) and judge:
                            if "强烈推荐" in judge:
                                level = "强烈推荐"
                                reading_recommendation = "强烈推荐精读"
                            elif "不推荐" in judge:
                                level = "不推荐"
                                reading_recommendation = "不推荐精读"
                            elif "谨慎" in judge:
                                level = "谨慎"
                                reading_recommendation = "谨慎精读"
                            elif "一般" in judge:
                                level = "一般"
                                reading_recommendation = "一般"
                            elif "推荐" in judge:
                                level = "推荐"
                                reading_recommendation = "推荐精读"
                    except Exception:
                        pass
                    if not level:
                        # 兜底：维持原默认
                        level = "一般"
                    if not reading_recommendation:
                        # 兜底：根据重要程度推断是否精读
                        if level in ["强烈推荐", "推荐"]:
                            reading_recommendation = "推荐精读"
                        elif level == "不推荐":
                            reading_recommendation = "不推荐精读"
                        else:
                            reading_recommendation = "一般"
                    
                    if text.endswith("---"):
                        text = text[:-3].rstrip() + f"\n论文重要程度: \"{level}\"\n是否精读: \"{reading_recommendation}\"\n---"
                except Exception:
                    pass
                
                # 清理列表字段中的None值
                list_fields = ["urls", "doi", "journal_or_conference", "year", "source_code"]
                text = self._clean_yaml_list(text, list_fields)
                
                return text
            return None

        except Exception as e:
            self.chatbot.append(["警告", f"生成 YAML 头失败: {str(e)}"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return None

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
            # 注入一次全文到上下文历史，后续多轮仅发送问题
            try:
                remembered = (
                    "请记住以下论文全文，后续所有问题仅基于此内容回答，不要重复输出原文：\n\n"
                    f"{self.paper_content}"
                )
                self.context_history = [remembered, "已接收并记住论文内容"]
            except Exception:
                self.context_history = []
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return True
        else:
            self.chatbot.append(["错误", "无法读取论文内容，请检查文件是否有效"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return False

    def _analyze_question(self, question: RFICQuestion) -> Generator:
        """分析单个问题 - 专门针对RF IC领域"""
        try:
            # 创建针对RF IC的分析提示（不再重复发送全文）
            prompt = f"""请基于已记住的射频集成电路论文全文，从RF IC专业角度回答问题：

问题：{question.question}

请从以下角度进行分析：
1. 技术深度：深入分析电路设计原理和技术细节
2. 工程价值：评估技术的实用性和产业化前景
3. 创新性：识别技术突破和创新点
4. 行业影响：分析对RF IC行业发展的意义

请保持专业性和技术准确性，使用RF IC领域的专业术语。"""

            # 使用单线程版本的请求函数
            response = yield from request_gpt_model_in_new_thread_with_ui_alive(
                inputs=prompt,
                inputs_show_user=question.question,  # 显示问题本身
                llm_kwargs=self.llm_kwargs,
                chatbot=self.chatbot,
                history=self.context_history or [],  # 复用上下文
                sys_prompt="你是一个专业的射频集成电路(RF IC)分析专家，具有深厚的电路设计、半导体工艺和无线通信系统知识。请从RF IC专业角度深入分析论文，使用准确的术语，提供有见地的技术评估。"
            )

            if response:
                self.results[question.id] = response
                # 记录本轮交互的输入与输出用于token估算
                try:
                    self._token_inputs.append(prompt)
                    self._token_outputs.append(response)
                except Exception:
                    pass
                return True
            return False

        except Exception as e:
            self.chatbot.append(["错误", f"分析问题时出错: {str(e)}"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return False

    def _generate_summary(self) -> Generator:
        """生成RF IC论文解读报告"""
        self.chatbot.append(["生成报告", "正在整合RF IC分析结果，生成专业解读报告..."])
        yield from update_ui(chatbot=self.chatbot, history=self.history)

        summary_prompt = """请基于以下对RF IC论文的各个方面的专业分析，生成一份全面的射频集成电路论文解读报告。

报告要求：
1. 突出RF IC技术特点和创新点
2. 强调电路设计的技术价值
3. 分析市场应用前景
4. 评估技术成熟度
5. 提供行业发展趋势洞察

请保持专业性和技术深度，适合RF IC工程师和研究人员阅读。"""

        for q in self.questions:
            if q.id in self.results:
                summary_prompt += f"\n\n关于{q.description}的专业分析:\n{self.results[q.id]}"

        try:
            # 使用单线程版本的请求函数，可以在前端实时显示生成结果
            response = yield from request_gpt_model_in_new_thread_with_ui_alive(
                inputs=summary_prompt,
                inputs_show_user="生成RF IC论文专业解读报告",
                llm_kwargs=self.llm_kwargs,
                chatbot=self.chatbot,
                history=[],
                sys_prompt="你是一个射频集成电路领域的资深专家，请将多个方面的专业分析整合为一份完整、深入、专业的RF IC论文解读报告。报告应当突出技术深度，体现工程价值，并对行业发展趋势提供专业洞察。"
            )

            if response:
                return response
            return "报告生成失败"

        except Exception as e:
            self.chatbot.append(["错误", f"生成报告时出错: {str(e)}"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return "报告生成失败: " + str(e)

    def save_report(self, report: str, paper_file_path: str = None) -> str:
        """保存RF IC分析报告，返回保存的文件路径"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 获取PDF文件名（不含扩展名）
        pdf_filename = "未知RF_IC论文"
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
            md_content = ""
            if self.yaml_header:
                md_content += self.yaml_header + "\n\n"
            md_content += f"# 射频集成电路论文专业解读报告\n\n"
            md_content += f"**分析时间**: {timestamp}\n"
            md_content += f"**论文文件**: {os.path.basename(paper_file_path) if paper_file_path else '未知'}\n\n"
            md_content += f"## 报告摘要\n\n{report}\n\n"
            md_content += f"## 详细技术分析\n\n"
            
            for q in self.questions:
                if q.id in self.results:
                    md_content += f"### {q.description}\n\n{self.results[q.id]}\n\n"

            # 追加 Token 估算结果
            try:
                stats = estimate_token_usage(self._token_inputs, self._token_outputs, self.llm_kwargs.get('llm_model', 'gpt-3.5-turbo'))
                if stats and stats.get('sum_total_tokens', 0) > 0:
                    md_content += (
                        "## Token 估算\n\n"
                        f"- 模型: {stats.get('model')}\n\n"
                        f"- 输入 tokens: {stats.get('sum_input_tokens', 0)}\n"
                        f"- 输出 tokens: {stats.get('sum_output_tokens', 0)}\n"
                        f"- 总 tokens: {stats.get('sum_total_tokens', 0)}\n\n"
                    )
            except Exception:
                pass

            result_file = write_history_to_file(
                history=[md_content],
                file_basename=f"{timestamp}_{pdf_filename}_RF_IC解读报告.md"
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
        """分析单篇RF IC论文主流程"""
        # 加载论文
        success = yield from self._load_paper(paper_path)
        if not success:
            return None

        # 分析关键问题 - 专门针对RF IC领域
        for question in self.questions:
            yield from self._analyze_question(question)

        # 生成总结报告
        final_report = yield from self._generate_summary()

        # 生成 YAML 头，与 Batch_Paper_Reading 一致流程
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
    base_save_dir = get_log_folder(get_user(chatbot), plugin_name='rf_ic_paper_download')
    save_dir = os.path.join(base_save_dir, f"rf_ic_papers_{timestamp}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = Path(save_dir)

    chatbot.append([f"下载RF IC论文", f"正在下载{'arXiv' if id_type == 'arxiv' else 'DOI'} {paper_id} 的论文..."])
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
            chatbot.append([f"下载成功", f"已成功下载RF IC论文: {os.path.basename(pdf_path)}"])
            update_ui(chatbot=chatbot, history=history)
            return pdf_path
        else:
            chatbot.append([f"下载失败", f"RF IC论文下载失败: {paper_id}"])
            update_ui(chatbot=chatbot, history=history)
            return None

    except Exception as e:
        chatbot.append([f"下载错误", f"下载RF IC论文时出错: {str(e)}"])
        update_ui(chatbot=chatbot, history=history)
        return None


@CatchException
def 批量射频集成电路论文速读(txt: str, llm_kwargs: Dict, plugin_kwargs: Dict, chatbot: List,
                     history: List, system_prompt: str, user_request: str):
    """主函数 - 批量射频集成电路论文速读"""
    # 初始化分析器
    chatbot.append(["函数插件功能及使用方式", "批量射频集成电路论文速读：专门针对RF IC领域的批量论文分析工具，为每篇RF IC论文生成专业的速读报告，适用于RF IC工程师和研究人员的快速技术评估。 <br><br>📋 使用方式：<br>1、输入包含多个RF IC相关PDF文件的文件夹路径<br>2、或者输入多个RF IC论文ID（DOI或arXiv ID），用逗号分隔<br>3、点击插件开始批量分析<br><br>🎯 专业分析领域：<br>- 射频前端电路（LNA、PA、混频器等）<br>- 频率合成器（VCO、PLL等）<br>- 无线通信系统集成<br>- 毫米波和太赫兹技术<br>- 低功耗和高效能设计"])
    yield from update_ui(chatbot=chatbot, history=history)

    paper_files = []

    # 检查输入是否包含论文ID（多个ID用逗号分隔）
    if ',' in txt:
        # 处理多个论文ID
        paper_ids = [id.strip() for id in txt.split(',') if id.strip()]
        chatbot.append(["检测到多个RF IC论文ID", f"检测到 {len(paper_ids)} 个RF IC论文ID，准备批量下载..."])
        yield from update_ui(chatbot=chatbot, history=history)

        for i, paper_id in enumerate(paper_ids):
            paper_info = extract_paper_id(paper_id)
            if paper_info:
                chatbot.append([f"下载RF IC论文 {i+1}/{len(paper_ids)}", f"正在下载 {'arXiv' if paper_info[0] == 'arxiv' else 'DOI'} ID: {paper_info[1]}..."])
                yield from update_ui(chatbot=chatbot, history=history)

                paper_file = download_paper_by_id(paper_info, chatbot, history)
                if paper_file:
                    paper_files.append(paper_file)
                else:
                    chatbot.append([f"下载失败", f"无法下载RF IC论文: {paper_id}"])
                    yield from update_ui(chatbot=chatbot, history=history)
            else:
                chatbot.append([f"ID格式错误", f"无法识别RF IC论文ID格式: {paper_id}"])
                yield from update_ui(chatbot=chatbot, history=history)
    else:
        # 检查单个论文ID
        paper_info = extract_paper_id(txt)
        if paper_info:
            # 单个论文ID
            chatbot.append(["检测到RF IC论文ID", f"检测到{'arXiv' if paper_info[0] == 'arxiv' else 'DOI'} ID: {paper_info[1]}，准备下载RF IC论文..."])
            yield from update_ui(chatbot=chatbot, history=history)

            paper_file = download_paper_by_id(paper_info, chatbot, history)
            if paper_file:
                paper_files.append(paper_file)
            else:
                report_exception(chatbot, history, a=f"下载RF IC论文失败", b=f"无法下载{'arXiv' if paper_info[0] == 'arxiv' else 'DOI'}论文: {paper_info[1]}")
                yield from update_ui(chatbot=chatbot, history=history)
                return
        else:
            # 检查输入路径
            if not os.path.exists(txt):
                report_exception(chatbot, history, a=f"批量解析RF IC论文: {txt}", b=f"找不到文件或无权访问: {txt}")
                yield from update_ui(chatbot=chatbot, history=history)
                return

            # 验证路径安全性
            user_name = chatbot.get_user()
            validate_path_safety(txt, user_name)

            # 查找所有论文文件
            paper_files = _find_paper_files(txt)

            if not paper_files:
                report_exception(chatbot, history, a=f"批量解析RF IC论文", b=f"在路径 {txt} 中未找到支持的论文文件")
                yield from update_ui(chatbot=chatbot, history=history)
                return

    yield from update_ui(chatbot=chatbot, history=history)

    # 开始批量分析
    if not paper_files:
        chatbot.append(["错误", "没有找到任何可分析的RF IC论文文件"])
        yield from update_ui(chatbot=chatbot, history=history)
        return

    chatbot.append(["开始批量RF IC分析", f"找到 {len(paper_files)} 篇RF IC论文，开始专业批量分析..."])
    yield from update_ui(chatbot=chatbot, history=history)

    # 创建批量RF IC分析器
    analyzer = BatchRFICAnalyzer(llm_kwargs, plugin_kwargs, chatbot, history, system_prompt)
    
    # 批量分析每篇RF IC论文
    successful_reports = []
    failed_papers = []
    
    for i, paper_file in enumerate(paper_files):
        try:
            chatbot.append([f"分析RF IC论文 {i+1}/{len(paper_files)}", f"正在专业分析: {os.path.basename(paper_file)}"])
            yield from update_ui(chatbot=chatbot, history=history)
            
            # 分析单篇RF IC论文
            saved_file = yield from analyzer.analyze_paper(paper_file)
            
            if saved_file:
                successful_reports.append((os.path.basename(paper_file), saved_file))
                chatbot.append([f"完成RF IC论文 {i+1}/{len(paper_files)}", f"成功分析并保存专业报告: {os.path.basename(saved_file)}"])
            else:
                failed_papers.append(os.path.basename(paper_file))
                chatbot.append([f"失败RF IC论文 {i+1}/{len(paper_files)}", f"分析失败: {os.path.basename(paper_file)}"])
            
            yield from update_ui(chatbot=chatbot, history=history)
            
        except Exception as e:
            failed_papers.append(os.path.basename(paper_file))
            chatbot.append([f"错误RF IC论文 {i+1}/{len(paper_files)}", f"分析出错: {os.path.basename(paper_file)} - {str(e)}"])
            yield from update_ui(chatbot=chatbot, history=history)

    # 生成批量RF IC分析总结
    summary = f"批量RF IC论文分析完成！\n\n"
    summary += f"📊 RF IC分析统计：\n"
    summary += f"- 总论文数：{len(paper_files)}\n"
    summary += f"- 成功分析：{len(successful_reports)}\n"
    summary += f"- 分析失败：{len(failed_papers)}\n\n"
    
    if successful_reports:
        summary += f"✅ 成功生成RF IC专业报告：\n"
        for paper_name, report_path in successful_reports:
            summary += f"- {paper_name} → {os.path.basename(report_path)}\n"
    
    if failed_papers:
        summary += f"\n❌ 分析失败的RF IC论文：\n"
        for paper_name in failed_papers:
            summary += f"- {paper_name}\n"
    
    summary += f"\n🎯 专业分析覆盖：\n"
    summary += f"- 电路架构与拓扑分析\n"
    summary += f"- 性能指标评估\n"
    summary += f"- 设计技术与工艺分析\n"
    summary += f"- 应用场景与市场定位\n"
    summary += f"- 技术挑战与创新点\n"
    summary += f"- 发展趋势与行业启示"

    chatbot.append(["批量RF IC分析完成", summary])
    yield from update_ui(chatbot=chatbot, history=history)
