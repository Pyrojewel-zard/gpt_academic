import os
import time
from dataclasses import dataclass
from typing import Dict, List, Generator
from crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive
from toolbox import update_ui, promote_file_to_downloadzone, write_history_to_file, CatchException, report_exception
from shared_utils.fastapi_server import validate_path_safety
from crazy_functions.paper_fns.paper_download import extract_paper_id, get_arxiv_paper, format_arxiv_id
from pathlib import Path
from datetime import datetime


@dataclass
class DeepReadQuestion:
    id: str
    question: str
    importance: int
    description: str
    domain: str  # "general" | "rf_ic" | "both"


class SimplePaperDetailAnalyzer:
    """单篇论文精读（轻量版本，沿用速读式每问一题策略，问题库沿用精读版提示）"""

    def __init__(self, llm_kwargs: Dict, plugin_kwargs: Dict, chatbot: List, history: List, system_prompt: str):
        self.llm_kwargs = llm_kwargs
        self.plugin_kwargs = plugin_kwargs
        self.chatbot = chatbot
        self.history = history
        self.system_prompt = system_prompt
        self.paper_content = ""
        self.results: Dict[str, str] = {}
        self.paper_file_path: str = None
        self.context_history: List[str] = []  # 一次性注入全文，后续每问仅发送题干

        # 提问问题库：与 undefine_paper_detail_reading 中保持一致（不改动题干）
        self.questions: List[DeepReadQuestion] = [
            DeepReadQuestion(
                id="problem_domain_and_motivation",
                description="问题域与动机分析",
                importance=5,
                domain="both",
                question=(
                    "【第一层：问题域理解】\n"
                    "请以学术教授视角分析研究背景与动机：\n"
                    "1) 论文要解决的核心问题与系统级重要性\n"
                    "2) 现有方法的根本性缺陷/局限\n"
                    "3) 本文方案的关键创新点及差异化\n"
                    "4) 对理论或工程应用的意义"
                ),
            ),
            DeepReadQuestion(
                id="theoretical_framework_and_contributions",
                description="理论框架与核心贡献",
                importance=5,
                domain="both",
                question=(
                    "【第二层：理论构建】\n"
                    "请系统梳理理论框架与核心贡献：\n"
                    "1) 理论/模型/等效变换及其关键假设\n"
                    "2) 核心贡献条目（按理论重要性排序）\n"
                    "3) 这些贡献如何针对性解决第一层缺陷\n"
                    "4) 适用范围与边界条件"
                ),
            ),
            DeepReadQuestion(
                id="method_design_and_technical_details",
                description="方法设计与技术细节（含复现与流程图）",
                importance=5,
                domain="both",
                question=(
                    "【第三层：方法与实现】\n"
                    "请从可复现角度拆解方法与实现细节，并给出1个流程图：\n"
                    "1) 核心算法/电路/系统的关键步骤与设计抉择\n"
                    "2) 关键符号/目标/推导要点（只保留必要环节）\n"
                    "3) 复现要点：关键超参数/资源与时间/实现难点及对策\n"
                    "4) 与现有方法的本质差异与权衡\n"
                    "5) 用```mermaid 给出一张流程图（flowchart TD或LR），体现整体流程与关键分支"
                ),
            ),
            DeepReadQuestion(
                id="experimental_validation_and_effectiveness",
                description="实验验证与有效性分析",
                importance=5,
                domain="both",
                question=(
                    "【第四层：实验与证据】\n"
                    "请评价实验设计与结果是否充分支撑主张：\n"
                    "1) 验证项是否针对性覆盖理论与方法关键点\n"
                    "2) 指标选择、对比方法与消融设计的合理性\n"
                    "3) 核心结果与主张的一致性（含统计显著性/可重复性）\n"
                    "4) 简述1-2条主要局限/风险与其影响"
                ),
            ),
            DeepReadQuestion(
                id="worth_reading_judgment",
                description="是否值得精读",
                importance=4,
                domain="both",
                question=(
                    "【重要性评估】\n"
                    "以教授视角给出精读建议（五选一）：强烈推荐/推荐/一般/谨慎/不推荐；并用1-2句说明理由。"
                ),
            ),
            DeepReadQuestion(
                id="rf_ic_circuit_architecture_detail",
                description="RF IC电路架构与工艺设计",
                importance=5,
                domain="rf_ic",
                question=(
                    "【RF IC：电路与工艺】\n"
                    "请聚焦电路架构与工艺器件的协同：\n"
                    "1) 核心电路模块架构（LNA/PA/混频/VCO/PLL等）\n"
                    "2) 工艺技术选择与器件优化策略\n"
                    "3) 工艺参数对射频性能的关键影响\n"
                    "4) 工艺-器件-电路的协同创新点"
                ),
            ),
            DeepReadQuestion(
                id="rf_ic_performance_and_methods",
                description="RF IC性能指标与设计方法",
                importance=5,
                domain="rf_ic",
                question=(
                    "【RF IC：性能与方法】\n"
                    "请从性能-功耗-面积(PPA)出发审视设计方法：\n"
                    "1) 关键指标（频率/带宽/增益/NF/线性度/效率等）与约束\n"
                    "2) 设计流程/方法学创新/EDA工具链策略\n"
                    "3) 版图/布线/射频优化与关键路径/瓶颈"
                ),
            ),
            DeepReadQuestion(
                id="rf_ic_manufacturing_market_analysis",
                description="RF IC制造、测试与市场分析",
                importance=4,
                domain="rf_ic",
                question=(
                    "【RF IC：制造与市场】\n"
                    "1) 制造工艺可行性/良率/成本与封装/可靠性\n"
                    "2) 射频测试策略/覆盖率与量产一致性\n"
                    "3) 应用场景/市场定位/商业化前景"
                ),
            ),
            DeepReadQuestion(
                id="presentation_summary_and_materials",
                description="演示材料与PPT摘要",
                importance=4,
                domain="both",
                question=(
                    "【演示材料】\n"
                    "请输出用于PPT的Markdown摘要：\n"
                    "# 总述（1行）\n- 一句话概括论文做了什么、为何有效\n\n"
                    "# 核心模块要点（3-5条）\n- 每条≤14字，概括输入→处理→输出与关键分支（RF IC强调电路/信号流/性能）\n\n"
                    "# 关键方法摘要（5-8条）\n- 每条≤16字，聚焦创新点与核心机制\n\n"
                    "# 应用与效果（2-3条，可选）\n- 场景/指标/收益\n\n"
                    "注意：仅输出上述Markdown结构，不嵌入代码。"
                ),
            ),
        ]
        self.questions.sort(key=lambda q: q.importance, reverse=True)

    def _load_paper(self, paper_path: str) -> Generator:
        from crazy_functions.doc_fns.text_content_loader import TextContentLoader
        yield from update_ui(chatbot=self.chatbot, history=self.history)

        self.paper_file_path = paper_path
        loader = TextContentLoader(self.chatbot, self.history)
        yield from loader.execute_single_file(paper_path)

        if len(self.history) >= 2 and self.history[-2]:
            self.paper_content = self.history[-2]
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

    def _ask(self, q: DeepReadQuestion) -> Generator:
        """每问一题，题干保持不变；上下文仅用一次性注入的全文。"""
        try:
            # 提问内容保持与原精读版一致（不改动题干）
            prompt = q.question

            # 使用简洁系统提示，避免增加负担
            sys_prompt = "你是一个专业的科研论文分析助手，请基于已记住的论文全文回答问题，保持客观和准确。"

            resp = yield from request_gpt_model_in_new_thread_with_ui_alive(
                inputs=prompt,
                inputs_show_user=q.question,
                llm_kwargs=self.llm_kwargs,
                chatbot=self.chatbot,
                history=self.context_history or [],
                sys_prompt=sys_prompt
            )
            if resp:
                self.results[q.id] = resp
                return True
            return False
        except Exception as e:
            self.chatbot.append(["错误", f"精读问题分析失败: {str(e)}"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return False

    def _generate_summary(self) -> Generator:
        """生成最终精读总结（轻量）"""
        self.chatbot.append(["生成报告", "正在整合精读结果，生成精读总结..."])
        yield from update_ui(chatbot=self.chatbot, history=self.history)

        summary_prompt = "请基于以下对论文各方面的精读分析，生成一份结构清晰、条理清楚的精读报告，总结关键观点与证据："
        for q in self.questions:
            if q.id in self.results:
                summary_prompt += f"\n\n关于{q.description}的分析:\n{self.results[q.id]}"

        try:
            resp = yield from request_gpt_model_in_new_thread_with_ui_alive(
                inputs=summary_prompt,
                inputs_show_user="生成论文精读总结报告",
                llm_kwargs=self.llm_kwargs,
                chatbot=self.chatbot,
                history=[],
                sys_prompt="你是科研论文精读专家，请将多方面分析整合为结构清晰的精读报告，避免重复原文。"
            )
            return resp or "报告生成失败"
        except Exception as e:
            self.chatbot.append(["错误", f"生成报告时出错: {str(e)}"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return "报告生成失败: " + str(e)

    def save_report(self, report: str) -> Generator:
        """保存精读报告"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        pdf_filename = "未知论文"
        if self.paper_file_path and os.path.exists(self.paper_file_path):
            pdf_filename = os.path.splitext(os.path.basename(self.paper_file_path))[0]
            import re
            pdf_filename = re.sub(r'[^\w\u4e00-\u9fff]', '_', pdf_filename)
            if len(pdf_filename) > 50:
                pdf_filename = pdf_filename[:50]

        try:
            md_content = f"# 论文精读报告\n\n{report}"
            for q in self.questions:
                if q.id in self.results:
                    md_content += f"\n\n## {q.description}\n\n{self.results[q.id]}"

            result_file = write_history_to_file(
                history=[md_content],
                file_basename=f"{timestamp}_{pdf_filename}_精读报告.md"
            )

            if result_file and os.path.exists(result_file):
                promote_file_to_downloadzone(result_file, chatbot=self.chatbot)
                self.chatbot.append(["保存成功", f"精读报告已保存至: {os.path.basename(result_file)}"])
                yield from update_ui(chatbot=self.chatbot, history=self.history)
            else:
                self.chatbot.append(["警告", "保存报告成功但找不到文件"])
                yield from update_ui(chatbot=self.chatbot, history=self.history)
        except Exception as e:
            self.chatbot.append(["警告", f"保存报告失败: {str(e)}"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)

    def analyze_paper(self, paper_path: str) -> Generator:
        ok = yield from self._load_paper(paper_path)
        if not ok:
            return

        for q in self.questions:
            yield from self._ask(q)

        report = yield from self._generate_summary()
        yield from update_ui(chatbot=self.chatbot, history=self.history)
        yield from self.save_report(report)


def _find_paper_file(path: str) -> str:
    """查找路径中的论文文件（简化版）"""
    if os.path.isfile(path):
        return path
    extensions = ["pdf", "docx", "doc", "txt", "md", "tex"]
    if os.path.isdir(path):
        try:
            for ext in extensions:
                p = os.path.join(path, f"paper.{ext}")
                if os.path.exists(p) and os.path.isfile(p):
                    return p
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path):
                    file_ext = file.split('.')[-1].lower() if '.' in file else ""
                    if file_ext in extensions:
                        return file_path
        except Exception:
            pass
    return None


def _download_paper_by_id(paper_info, chatbot, history) -> str:
    """与速读版一致的下载逻辑，支持 arxiv/DOI"""
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
def 单篇论文精读(txt: str, llm_kwargs: Dict, plugin_kwargs: Dict, chatbot: List,
          history: List, system_prompt: str, user_request: str):
    """主函数 - 单篇论文精读（轻量、问题库与精读版一致，提问方式与速读一致）"""
    chatbot.append(["函数插件功能及使用方式", "单篇论文精读：沿用精读问题库，但采用轻量逐问策略。输入PDF路径、目录或 DOI/arXiv ID 后开始分析。"])
    yield from update_ui(chatbot=chatbot, history=history)

    paper_file = None

    paper_info = extract_paper_id(txt)
    if paper_info:
        chatbot.append(["检测到论文ID", f"检测到{'arXiv' if paper_info[0] == 'arxiv' else 'DOI'} ID: {paper_info[1]}，准备下载论文..."])
        yield from update_ui(chatbot=chatbot, history=history)
        paper_file = _download_paper_by_id(paper_info, chatbot, history)
        if not paper_file:
            report_exception(chatbot, history, a="下载论文失败", b=f"无法下载论文: {paper_info[1]}")
            yield from update_ui(chatbot=chatbot, history=history)
            return
    else:
        if not os.path.exists(txt):
            report_exception(chatbot, history, a=f"解析论文: {txt}", b=f"找不到文件或无权访问: {txt}")
            yield from update_ui(chatbot=chatbot, history=history)
            return
        user_name = chatbot.get_user()
        validate_path_safety(txt, user_name)
        paper_file = _find_paper_file(txt)
        if not paper_file:
            report_exception(chatbot, history, a=f"解析论文", b=f"在路径 {txt} 中未找到支持的论文文件")
            yield from update_ui(chatbot=chatbot, history=history)
            return

    yield from update_ui(chatbot=chatbot, history=history)

    # 类型检查与转换
    chatbot.append(["文件类型检查", f"paper_file类型: {type(paper_file)}, 值: {paper_file}"])
    yield from update_ui(chatbot=chatbot, history=history)
    chatbot.pop()
    if paper_file is not None and not isinstance(paper_file, str):
        try:
            paper_file = str(paper_file)
        except:
            report_exception(chatbot, history, a=f"类型错误", b=f"论文路径不是有效的字符串: {type(paper_file)}")
            yield from update_ui(chatbot=chatbot, history=history)
            return

    chatbot.append(["开始精读", f"正在精读论文: {os.path.basename(paper_file)}"])
    yield from update_ui(chatbot=chatbot, history=history)

    analyzer = SimplePaperDetailAnalyzer(llm_kwargs, plugin_kwargs, chatbot, history, system_prompt)
    yield from analyzer.analyze_paper(paper_file)


