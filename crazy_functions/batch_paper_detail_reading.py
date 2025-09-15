import os
import re
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Generator
from crazy_functions.Batch_Paper_Reading import estimate_token_usage
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
        self.context_history: List[str] = []  # 与LLM共享的上下文（每篇论文注入一次全文）
        # 统计用：记录每次LLM交互的输入与输出
        self._token_inputs: List[str] = []
        self._token_outputs: List[str] = []

        # 精读维度（递进式深入分析，从宏观到微观，从理论到实践）
        self.questions: List[DeepReadQuestion] = [
            # 第一层：问题域与动机分析
            DeepReadQuestion(
                id="problem_domain_and_motivation",
                description="问题域与动机分析",
                importance=5,
                question=(
                    "【第一层：问题域理解】\n"
                    "请深入分析论文的研究背景与动机：\n"
                    "1) 论文要解决的核心问题是什么？该问题在领域中的重要性如何？\n"
                    "2) 现有方法存在哪些根本性缺陷或局限性？\n"
                    "3) 论文提出的解决思路的独特性和创新性体现在哪里？\n"
                    "4) 该研究对理论发展或实际应用的意义是什么？"
                ),
            ),
            
            # 第二层：理论框架与核心贡献
            DeepReadQuestion(
                id="theoretical_framework_and_contributions",
                description="理论框架与核心贡献",
                importance=5,
                question=(
                    "【第二层：理论构建】\n"
                    "基于前面对问题域的理解，请深入分析论文的理论框架：\n"
                    "1) 论文建立了什么样的理论框架或数学模型？\n"
                    "2) 核心贡献有哪些？请按理论重要性排序并说明每个贡献的独特价值\n"
                    "3) 这些贡献如何解决第一层中识别的现有方法缺陷？\n"
                    "4) 理论框架的适用范围和边界条件是什么？"
                ),
            ),
            
            # 第三层：方法设计与技术细节
            DeepReadQuestion(
                id="method_design_and_technical_details",
                description="方法设计与技术细节",
                importance=5,
                question=(
                    "【第三层：技术实现】\n"
                    "基于前面的理论框架，请深入分析具体的技术实现：\n"
                    "1) 核心算法的设计思路和关键步骤是什么？\n"
                    "2) 关键符号定义、损失函数/目标函数、以及主要定理/引理的推导过程\n"
                    "3) 算法的时间复杂度、空间复杂度以及收敛性分析\n"
                    "4) 实现中的关键技术难点和解决方案\n"
                    "5) 与现有方法在技术层面的本质区别是什么？"
                ),
            ),
            
            # 第四层：实验验证与有效性分析
            DeepReadQuestion(
                id="experimental_validation_and_effectiveness",
                description="实验验证与有效性分析",
                importance=5,
                question=(
                    "【第四层：实验验证】\n"
                    "基于前面的技术设计，请分析实验如何验证方法的有效性：\n"
                    "1) 实验设计如何验证前面提出的理论贡献？\n"
                    "2) 数据集选择、评估指标和对比方法的合理性分析\n"
                    "3) 主要实验结果是否支持论文的核心主张？\n"
                    "4) 消融实验揭示了哪些关键因素和交互效应？\n"
                    "5) 实验结果的统计显著性和可重复性如何？"
                ),
            ),
            
            # 第五层：假设条件与局限性分析
            DeepReadQuestion(
                id="assumptions_limitations_and_threats",
                description="假设条件与局限性分析",
                importance=4,
                question=(
                    "【第五层：批判性分析】\n"
                    "基于前面的全面分析，请进行批判性思考：\n"
                    "1) 论文的显式和隐式假设有哪些？这些假设的合理性如何？\n"
                    "2) 在什么条件下方法可能失效？现实应用中的潜在风险是什么？\n"
                    "3) 实验设计的局限性和可能的误导性结论\n"
                    "4) 作者未充分讨论但可能影响方法有效性的因素\n"
                    "5) 方法的可扩展性和泛化能力如何？"
                ),
            ),
            
            # 第六层：复现指南与工程实现
            DeepReadQuestion(
                id="reproduction_guide_and_engineering",
                description="复现指南与工程实现",
                importance=5,
                question=(
                    "【第六层：工程复现】\n"
                    "基于前面的技术分析，请提供复现指导：\n"
                    "1) 复现所需的数据集、预训练模型和依赖资源\n"
                    "2) 关键超参数及其调优策略（不涉及具体数值）\n"
                    "3) 训练和评估流程的关键步骤\n"
                    "4) 硬件资源需求（GPU/CPU/内存/存储）和时间成本估算\n"
                    "5) 可能遇到的实现难点和解决方案\n"
                    "6) 开源代码的可用性和许可证情况"
                ),
            ),
            
            # 第七层：流程图与架构设计
            DeepReadQuestion(
                id="flowcharts_and_architecture",
                description="流程图与架构设计",
                importance=4,
                question=(
                    "【第七层：架构可视化】\n"
                    "基于前面的技术分析，请绘制核心流程图：\n"
                    "要求：\n"
                    "1) 每个流程图使用 Mermaid 语法，代码块需以 ```mermaid 开始，以 ``` 结束\n"
                    "2) 推荐使用 flowchart TD 或 LR，节点需概括关键步骤/子模块\n"
                    "3) 每个流程图前以一句话标明模块/阶段名称\n"
                    "4) 格式约束：\n"
                    "   - 节点名用引号包裹，如 [\"节点名\"] 或 (\"节点名\")\n"
                    "   - 箭头标签采用 |\"标签名\"| 形式\n"
                    "5) 重点展示：整体架构、核心算法流程、数据流向、关键决策点"
                ),
            ),
            
            # 第八层：影响评估与未来展望
            DeepReadQuestion(
                id="impact_assessment_and_future_directions",
                description="影响评估与未来展望",
                importance=3,
                question=(
                    "【第八层：影响与展望】\n"
                    "基于前面的全面分析，请评估研究的影响和前景：\n"
                    "1) 该研究对学术领域的短期和长期影响\n"
                    "2) 潜在的产业应用价值和商业化前景\n"
                    "3) 可能引发的后续研究方向\n"
                    "4) 存在的伦理问题或社会影响\n"
                    "5) 改进和扩展的具体建议"
                ),
            ),
            
            # 第九层：执行摘要与要点总结
            DeepReadQuestion(
                id="executive_summary_and_key_points",
                description="执行摘要与要点总结",
                importance=5,
                question=(
                    "【第九层：要点总结】\n"
                    "基于前面八层的深入分析，请给出精炼的执行摘要：\n"
                    "格式要求（Markdown，不包含代码）：\n"
                    "## 核心价值\n"
                    "- 一句话概括方法的核心价值\n"
                    "\n"
                    "## 技术要点\n"
                    "- 3-5条关键技术要点（输入/处理/输出）\n"
                    "\n"
                    "## 复现要点\n"
                    "- 3-5条复现关键信息（数据/参数/资源/时间）\n"
                    "\n"
                    "## 适用场景\n"
                    "- 2-3条典型应用场景\n"
                    "\n"
                    "## 注意事项\n"
                    "- 2-3条重要限制或注意事项"
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
                    rebuilt = ', '.join([f'\"{k}\"' for k in merged])
                    text = re.sub(r"^keywords:\s*\[(.*?)\]\s*$", f"keywords: [{rebuilt}]", text, flags=re.MULTILINE)
                # 注入“归属”二级分类到 YAML 头（仅写入分类路径本身，并用引号包裹）
                try:
                    if getattr(self, 'secondary_category', None):
                        escaped = self.secondary_category.replace('\"', '\\\"')
                        if text.endswith("---"):
                            text = text[:-3].rstrip() + f"\nsecondary_category: \"{escaped}\"\n---"
                except Exception:
                    pass
                # 基于 worth_reading_judgment 提取中文"论文重要程度"和"是否精读"，若缺失回退默认
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
        yield from update_ui(chatbot=self.chatbot, history=self.history)
        self.paper_file_path = paper_path
        loader = TextContentLoader(self.chatbot, self.history)
        yield from loader.execute_single_file(paper_path)
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
        self.chatbot.append(["错误", "无法读取论文内容，请检查文件是否有效"])
        yield from update_ui(chatbot=self.chatbot, history=self.history)
        return False

    def _ask(self, q: DeepReadQuestion) -> Generator:
        try:
            # 构建递进式分析的上下文
            context_parts = [
                "请基于已记住的论文全文进行递进式精读分析，并严格围绕问题作答。\n"
                "注意：请避免提供任何代码、伪代码、命令行或具体实现细节；"
                "若输出流程图，须使用 ```mermaid 代码块，其余回答保持自然语言。\n"
            ]
            
            # 添加前面分析的结果作为上下文
            if self.results:
                context_parts.append("\n【前面分析的关键发现】")
                for prev_q in self.questions:
                    if prev_q.id in self.results and prev_q.id != q.id:
                        # 只添加前面已分析的问题结果
                        if any(prev_q.id == existing_id for existing_id in self.results.keys()):
                            context_parts.append(f"\n{prev_q.description}：{self.results[prev_q.id][:300]}...")
            
            context_parts.append(f"\n\n【当前分析任务】\n{q.question}")
            
            prompt = "".join(context_parts)
            
            resp = yield from request_gpt_model_in_new_thread_with_ui_alive(
                inputs=prompt,
                inputs_show_user=q.question,
                llm_kwargs=self.llm_kwargs,
                chatbot=self.chatbot,
                history=self.context_history or [],
                sys_prompt=(
                    "你是资深研究员，进行递进式深度分析。每个问题都基于前面的分析结果进行深入。"
                    "输出以概念与方法论层面为主，不包含任何代码或伪代码。"
                    "如涉及Mermaid流程图，请使用```mermaid 包裹并保持语法正确，其余保持自然语言。"
                    "注意保持分析的连贯性和递进性。"
                ),
            )
            if resp:
                self.results[q.id] = resp
                # 记录本轮交互的输入与输出用于token估算
                try:
                    self._token_inputs.append(prompt)
                    self._token_outputs.append(resp)
                except Exception:
                    pass
                return True
            return False
        except Exception as e:
            self.chatbot.append(["错误", f"精读问题分析失败: {str(e)}"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return False

    def _generate_report(self) -> Generator:
        self.chatbot.append(["生成报告", "正在整合递进式精读结果，生成深度技术报告..."])
        yield from update_ui(chatbot=self.chatbot, history=self.history)

        prompt = (
            "请将以下递进式精读分析整理为完整的技术报告。"
            "报告应体现分析的递进逻辑：从问题域理解→理论构建→技术实现→实验验证→批判分析→工程复现→架构可视化→影响评估→要点总结。"
            "层次清晰，突出核心思想与实验设计要点，不包含任何代码/伪代码/命令行。"
            "若包含```mermaid 代码块，请原样保留。\n\n"
            "【递进式分析结果】"
        )
        
        # 按照递进顺序组织分析结果
        layer_order = [
            "problem_domain_and_motivation",
            "theoretical_framework_and_contributions", 
            "method_design_and_technical_details",
            "experimental_validation_and_effectiveness",
            "assumptions_limitations_and_threats",
            "reproduction_guide_and_engineering",
            "flowcharts_and_architecture",
            "impact_assessment_and_future_directions",
            "executive_summary_and_key_points"
        ]
        
        for layer_id in layer_order:
            for q in self.questions:
                if q.id == layer_id and q.id in self.results:
                    prompt += f"\n\n## {q.description}\n{self.results[q.id]}"
                    break

        resp = yield from request_gpt_model_in_new_thread_with_ui_alive(
            inputs=prompt,
            inputs_show_user="生成递进式论文精读技术报告",
            llm_kwargs=self.llm_kwargs,
            chatbot=self.chatbot,
            history=[],
            sys_prompt=(
                "以递进式深度分析为主线组织报告：体现从宏观到微观、从理论到实践的完整分析链条。"
                "每个部分都要与前面的分析形成逻辑关联，突出递进关系。"
                "以工程复现为目标，背景极简，方法与实现细节充分，条理分明，包含必要的清单与步骤。"
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
        if "executive_summary_and_key_points" in self.results:
            parts.append(f"\n\n## 执行级摘要\n\n{self.results['executive_summary_and_key_points']}")
        if "flowcharts_and_architecture" in self.results:
            parts.append(f"\n\n## 核心流程图\n\n{self.results['flowcharts_and_architecture']}")
        # 追加其余维度
        for q in self.questions:
            if q.id in self.results and q.id not in {"executive_summary_and_key_points", "flowcharts_and_architecture"}:
                parts.append(f"\n\n## {q.description}\n\n{self.results[q.id]}")

        # 追加 Token 估算结果
        try:
            stats = estimate_token_usage(self._token_inputs, self._token_outputs, self.llm_kwargs.get('llm_model', 'gpt-3.5-turbo'))
            if stats and stats.get('sum_total_tokens', 0) > 0:
                parts.append(
                    "\n\n## Token 估算\n\n"
                    f"- 模型: {stats.get('model')}\n\n"
                    f"- 输入 tokens: {stats.get('sum_input_tokens', 0)}\n"
                    f"- 输出 tokens: {stats.get('sum_output_tokens', 0)}\n"
                    f"- 总 tokens: {stats.get('sum_total_tokens', 0)}\n"
                )
        except Exception:
            pass

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


