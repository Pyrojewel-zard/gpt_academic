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


def estimate_token_usage(inputs: List[str], outputs: List[str], llm_model: str) -> Dict:
    """估算一组交互的输入/输出token消耗（与统一速读实现保持一致接口）。"""
    try:
        from request_llms.bridge_all import model_info  # 动态导入，避免循环依赖
        def _estimate_tokens(text: str) -> int:
            try:
                cnt_fn = model_info.get(llm_model, {}).get("token_cnt", None)
                if cnt_fn is None:
                    cnt_fn = model_info["gpt-3.5-turbo"]["token_cnt"]
                return int(cnt_fn(text or ""))
            except Exception:
                return len(text or "")
    except Exception:
        def _estimate_tokens(text: str) -> int:
            return len(text or "")

    n = max(len(inputs or []), len(outputs or []))
    items = []
    sum_in = 0
    sum_out = 0
    for i in range(n):
        inp = inputs[i] if i < len(inputs) else ""
        out = outputs[i] if i < len(outputs) else ""
        ti = _estimate_tokens(inp)
        to = _estimate_tokens(out)
        items.append({
            'input_tokens': ti,
            'output_tokens': to,
            'total_tokens': ti + to,
        })
        sum_in += ti
        sum_out += to
    return {
        'model': llm_model,
        'items': items,
        'sum_input_tokens': sum_in,
        'sum_output_tokens': sum_out,
        'sum_total_tokens': sum_in + sum_out,
    }

@dataclass
class DeepReadQuestion:
    """论文精读问题项"""
    id: str
    question: str
    importance: int
    description: str
    domain: str  # 适用领域 ("general", "rf_ic", "both")


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
        self.paper_domain = "general"  # 论文领域分类
        self.context_history: List[str] = []  # 与LLM共享的上下文（每篇论文注入一次全文）
        # 统计用：记录每次LLM交互的输入与输出
        self._token_inputs: List[str] = []
        self._token_outputs: List[str] = []
        # Token管理相关
        self._max_context_tokens = 15000  # 最大上下文token数
        self._token_usage_stats = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'interaction_count': 0
        }

        # 精读维度（递进式深入分析，支持领域自适应）
        self.questions: List[DeepReadQuestion] = [
            # 通用问题（适用于所有论文）
            # 第一层：问题域与动机分析
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
            
            # 第二层：理论框架与核心贡献
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
            
            # 第三层：方法设计（含复现与流程图）
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
            
            # 第四层：实验验证与有效性分析
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
            
            # 重要性与是否值得精读（用于生成推荐阅读等级和星级）
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
            
            # RF IC专用：电路架构与工艺设计（合并版）
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
            
            # RF IC专用：性能指标与设计方法（合并版）
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
            
            # RF IC专用：制造、测试与市场（合并版）
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
            
            # PPT摘要与演示材料（保留一个）
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

    def _classify_paper_domain(self) -> Generator:
        """使用LLM对论文进行主题分类，判断是否为RF IC相关论文"""
        try:
            classification_prompt = f"""请分析以下论文内容，判断其是否属于射频集成电路(RF IC)领域：

论文内容片段：
{self.paper_content[:2000]}...

请根据以下标准进行判断：
1. 如果论文涉及射频前端电路（LNA、PA、混频器、VCO、PLL等）
2. 如果论文涉及无线通信系统集成、毫米波技术、太赫兹技术
3. 如果论文涉及射频电路设计、半导体工艺在射频应用
4. 如果论文涉及射频性能指标（噪声系数、线性度、效率等）
5. 如果论文涉及到使用ML或者一系列EDA工具，涉及人工智能，那么就是GENERAL，即所有AI+RFIC也是GENERAL

请只回答："RF_IC" 或 "GENERAL"，不要其他内容。"""

            response = yield from request_gpt_model_in_new_thread_with_ui_alive(
                inputs=classification_prompt,
                inputs_show_user="正在分析论文主题分类...",
                llm_kwargs=self.llm_kwargs,
                chatbot=self.chatbot,
                history=[],
                sys_prompt="你是一个专业的论文分类助手，请根据论文内容准确判断其所属领域。"
            )

            if response and isinstance(response, str):
                response = response.strip().upper()
                if "RF_IC" in response:
                    self.paper_domain = "rf_ic"
                    self.chatbot.append(["主题分类", "检测到RF IC相关论文，将使用专业RF IC精读分析策略"])
                else:
                    self.paper_domain = "general"
                    self.chatbot.append(["主题分类", "检测到通用论文，将使用通用精读分析策略"])
            else:
                self.paper_domain = "general"
                self.chatbot.append(["主题分类", "无法确定主题，使用通用精读分析策略"])

            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return True

        except Exception as e:
            self.paper_domain = "general"
            self.chatbot.append(["分类错误", f"主题分类失败，使用通用策略: {str(e)}"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return False

    def _get_domain_specific_questions(self) -> List[DeepReadQuestion]:
        """根据论文领域获取相应的问题列表"""
        if self.paper_domain == "rf_ic":
            # RF IC论文：包含RF IC专用问题和核心通用问题，排除通用版流程图和PPT摘要（使用合并版）
            excluded_ids = set()  # 不需要排除了，因为已经合并
            return [q for q in self.questions if q.domain == "rf_ic" or (q.domain == "both")]
        else:
            # 通用论文：只包含通用问题，排除RF IC专用问题
            return [q for q in self.questions if q.domain in ["both", "general"]]

    def _get_domain_specific_system_prompt(self) -> str:
        """根据论文领域获取相应的系统提示"""
        if self.paper_domain == "rf_ic":
            return """你是一个专业的射频集成电路(RF IC)分析专家，具有深厚的电路设计、半导体工艺和无线通信系统知识。请从RF IC专业角度深入分析论文，使用准确的术语，提供有见地的技术评估。"""
        else:
            return """你是一个专业的科研论文分析助手，进行递进式深度分析。每个问题都基于前面的分析结果进行深入。输出以概念与方法论层面为主，不包含任何代码或伪代码。如涉及Mermaid流程图，请使用```mermaid 包裹并保持语法正确，其余保持自然语言。注意保持分析的连贯性和递进性。"""

    def _get_domain_config(self) -> Dict:
        """获取当前论文领域的配置"""
        if self.paper_domain == "rf_ic":
            return {
                "label": "RF IC专业",
                "result_label": "【递进式RF IC分析结果】",
                "report_title": "射频集成电路论文专业精读报告",
                "file_prefix": "RF_IC",
            }
        else:
            return {
                "label": "通用",
                "result_label": "【递进式分析结果】",
                "report_title": "论文精读技术报告",
                "file_prefix": "通用",
            }

    def _estimate_tokens(self, text: str) -> int:
        """估算文本的token数量（粗略估算：1 token ≈ 4 字符）"""
        if not text:
            return 0
        return len(text) // 4


    def _build_progressive_context(self, current_q: DeepReadQuestion) -> List[str]:
        """构建递进式分析的上下文，智能引用前面的分析结果"""
        context_parts = []
        
        if not self.results:
            return context_parts
        
        # 定义问题的重要性和依赖关系
        question_importance = {
            "problem_domain_and_motivation": 5,
            "theoretical_framework_and_contributions": 5,
            "method_design_and_technical_details": 5,
            "experimental_validation_and_effectiveness": 5,
            "worth_reading_judgment": 4,
            "rf_ic_circuit_architecture_detail": 5,
            "rf_ic_performance_and_methods": 5,
            "rf_ic_manufacturing_market_analysis": 4,
            "presentation_summary_and_materials": 4,
        }
        
        # 根据当前问题的重要性，智能选择引用的前面分析
        current_importance = question_importance.get(current_q.id, 3)
        
        # 收集前面分析的关键发现
        key_findings = []
        for prev_q in self.questions:
            if prev_q.id in self.results and prev_q.id != current_q.id:
                prev_importance = question_importance.get(prev_q.id, 3)
                # 只引用重要性较高或与当前问题相关的前面分析
                if prev_importance >= current_importance or self._is_related_analysis(prev_q.id, current_q.id):
                    key_findings.append((prev_q.description, self.results[prev_q.id]))
        
        if key_findings:
            context_parts.append("\n【前面分析的关键发现】")
            
            # 根据当前问题类型，智能组织前面分析的结果
            if current_q.id in ["theoretical_framework_and_contributions", "method_design_and_technical_details"]:
                # 理论和方法问题，重点引用问题域分析
                for desc, result in key_findings:
                    if "问题域" in desc or "动机" in desc:
                        context_parts.append(f"\n{desc}：{result[:400]}...")
            elif current_q.id in ["experimental_validation_and_effectiveness"]:
                # 实验验证问题，重点引用理论和方法分析
                for desc, result in key_findings:
                    if any(keyword in desc for keyword in ["理论", "方法", "技术", "算法"]):
                        context_parts.append(f"\n{desc}：{result[:400]}...")
            elif current_q.id in ["worth_reading_judgment"]:
                # 批判分析问题，引用所有前面的分析
                for desc, result in key_findings:
                    context_parts.append(f"\n{desc}：{result[:300]}...")
            else:
                # 其他问题，引用最重要的前面分析
                for desc, result in key_findings[:3]:  # 最多引用3个
                    context_parts.append(f"\n{desc}：{result[:300]}...")
        
        return context_parts

    def _is_related_analysis(self, prev_id: str, current_id: str) -> bool:
        """判断两个分析问题是否相关"""
        # 定义问题间的依赖关系
        dependencies = {
            "theoretical_framework_and_contributions": ["problem_domain_and_motivation"],
            "method_design_and_technical_details": ["problem_domain_and_motivation", "theoretical_framework_and_contributions"],
            "experimental_validation_and_effectiveness": ["theoretical_framework_and_contributions", "method_design_and_technical_details"],
            "worth_reading_judgment": ["experimental_validation_and_effectiveness"],
            "rf_ic_circuit_architecture_detail": ["method_design_and_technical_details"],
            "rf_ic_performance_and_methods": ["method_design_and_technical_details", "rf_ic_circuit_architecture_detail"],
            "rf_ic_manufacturing_market_analysis": ["rf_ic_performance_and_methods"],
            "presentation_summary_and_materials": ["method_design_and_technical_details", "experimental_validation_and_effectiveness"],
        }
        
        return prev_id in dependencies.get(current_id, [])

    def _update_token_stats(self, input_text: str, output_text: str):
        """更新token使用统计"""
        try:
            input_tokens = self._estimate_tokens(input_text)
            output_tokens = self._estimate_tokens(output_text)
            
            self._token_usage_stats['total_input_tokens'] += input_tokens
            self._token_usage_stats['total_output_tokens'] += output_tokens
            self._token_usage_stats['interaction_count'] += 1
            
            # 如果单次交互的token数过多，给出警告
            total_tokens = input_tokens + output_tokens
            if total_tokens > 2000:  # 单次交互超过2000 tokens
                self.chatbot.append(["Token警告", f"本次交互使用了 {total_tokens} tokens，建议检查输入长度"])
                
        except Exception:
            pass

    def _get_token_usage_summary(self) -> str:
        """获取token使用摘要"""
        try:
            stats = self._token_usage_stats
            total_tokens = stats['total_input_tokens'] + stats['total_output_tokens']
            avg_tokens = total_tokens / max(stats['interaction_count'], 1)
            
            summary = f"""
## Token 使用统计
- 总交互次数: {stats['interaction_count']}
- 总输入 tokens: {stats['total_input_tokens']}
- 总输出 tokens: {stats['total_output_tokens']}
- 总 tokens: {total_tokens}
- 平均每次交互: {avg_tokens:.1f} tokens
"""
            return summary
        except Exception:
            return "Token统计信息获取失败"

    def _check_token_limits(self, input_text: str) -> bool:
        """检查是否接近token限制"""
        try:
            current_tokens = self._estimate_tokens(input_text)
            context_tokens = sum(self._estimate_tokens(text) for text in self.context_history)
            total_estimated = current_tokens + context_tokens
            
            if total_estimated > self._max_context_tokens * 0.9:  # 接近90%限制
                self.chatbot.append(["Token警告", f"预计将使用 {total_estimated} tokens，接近限制 {self._max_context_tokens}"])
                return True
            return False
        except Exception:
            return False

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

            if isinstance(yaml_str, str):
                # 预处理：去除可能的代码围栏，并提取 --- ... --- 块
                raw = yaml_str.strip()
                # 去除三引号代码块
                m_code = re.search(r"^```[a-zA-Z]*\n([\s\S]*?)\n```$", raw, flags=re.MULTILINE)
                if m_code:
                    raw = m_code.group(1).strip()
                # 提取 --- ... ---
                if raw.count('---') >= 2:
                    first = raw.find('---')
                    last = raw.rfind('---')
                    text = raw[first:last+3].strip()
                else:
                    # 若无分隔符，则包裹为 Front Matter
                    text = f"---\n{raw}\n---"

                # 规范化关键词列表
                m = re.search(r"^keywords:\s*\[(.*?)\]\s*$", text, flags=re.MULTILINE)
                if m:
                    inner = m.group(1).strip()
                    raw_list = [x.strip().strip('\"\'\'') for x in inner.split(',') if x.strip()]
                    merged, _ = self._merge_keywords_with_db(raw_list)
                    rebuilt = ', '.join([f'\"{k}\"' for k in merged])
                    text = re.sub(r"^keywords:\s*\[(.*?)\]\s*$", f"keywords: [{rebuilt}]", text, flags=re.MULTILINE)
                # 注入"归属"二级分类到 YAML 头（仅写入分类路径本身，并用引号包裹）
                try:
                    if getattr(self, 'secondary_category', None):
                        escaped = self.secondary_category.replace('\"', '\\\"')
                        if text.endswith("---"):
                            text = text[:-3].rstrip() + f"\nsecondary_category: \"{escaped}\"\n---"
                except Exception:
                    pass
                # 基于 worth_reading_judgment 建立统一的精读建议→星级评分映射
                try:
                    # 定义精读建议到星级评分的唯一映射关系
                    reading_to_stars_mapping = {
                        "强烈推荐": {"level": "强烈推荐", "stars": "⭐⭐⭐⭐⭐", "reading": "强烈推荐精读"},
                        "推荐": {"level": "推荐", "stars": "⭐⭐⭐⭐", "reading": "推荐精读"},
                        "一般": {"level": "一般", "stars": "⭐⭐⭐", "reading": "一般"},
                        "谨慎": {"level": "谨慎", "stars": "⭐⭐", "reading": "谨慎精读"},
                        "不推荐": {"level": "不推荐", "stars": "⭐", "reading": "不推荐精读"}
                    }
                    
                    # 从worth_reading_judgment中提取精读建议
                    judge = self.results.get("worth_reading_judgment", "")
                    reading_level = "一般"  # 默认值
                    
                    if isinstance(judge, str) and judge:
                        for key in reading_to_stars_mapping.keys():
                            if key in judge:
                                reading_level = key
                                break
                    
                    # 获取映射结果
                    mapping_result = reading_to_stars_mapping[reading_level]
                    level = mapping_result["level"]
                    stars = mapping_result["stars"]
                    reading_recommendation = mapping_result["reading"]
                    
                    # 更新YAML中的stars字段
                    if re.search(r"^stars:\s*\[(.*?)\]\s*$", text, flags=re.MULTILINE):
                        text = re.sub(r"^stars:\s*\[(.*?)\]\s*$", f"stars: [\"{stars}\"]", text, flags=re.MULTILINE)
                    elif re.search(r"^stars:\s*.*$", text, flags=re.MULTILINE):
                        text = re.sub(r"^stars:\s*.*$", f"stars: [\"{stars}\"]", text, flags=re.MULTILINE)
                    else:
                        if text.endswith("---"):
                            text = text[:-3].rstrip() + f"\nstars: [\"{stars}\"]\n---"
                    
                    # 添加论文重要程度和是否精读字段
                    if text.endswith("---"):
                        text = text[:-3].rstrip() + f"\n论文重要程度: \"{level}\"\n是否精读: \"{reading_recommendation}\"\n---"
                except Exception:
                    pass
                
                # 清理列表字段中的None值
                list_fields = ["urls", "doi", "journal_or_conference", "year", "source_code"]
                text = self._clean_yaml_list(text, list_fields)
                
                # 强制设置 read_status 为 未阅读（无论模型如何返回）
                try:
                    if re.search(r"^read_status\s*:", text, flags=re.MULTILINE):
                        text = re.sub(r"^read_status\s*:.*$", 'read_status: "未阅读"', text, flags=re.MULTILINE)
                    else:
                        if text.endswith("---"):
                            text = text[:-3].rstrip() + '\n' + 'read_status: "未阅读"' + '\n---'
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
            
            # 智能构建递进式上下文
            context_parts.extend(self._build_progressive_context(q))
            
            context_parts.append(f"\n\n【当前分析任务】\n{q.question}")
            
            prompt = "".join(context_parts)
            
            # 检查token限制
            self._check_token_limits(prompt)
            
            # 获取领域特定的系统提示
            sys_prompt = self._get_domain_specific_system_prompt()
            
            resp = yield from request_gpt_model_in_new_thread_with_ui_alive(
                inputs=prompt,
                inputs_show_user=q.question,
                llm_kwargs=self.llm_kwargs,
                chatbot=self.chatbot,
                history=self.context_history or [],
                sys_prompt=sys_prompt,
            )
            if resp:
                self.results[q.id] = resp
                # 将问答对累积到历史记录中，实现上下文连贯性
                self.context_history.extend([prompt, resp])
                
                # 更新token统计
                self._update_token_stats(prompt, resp)
                
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
        domain_label = self._get_domain_config()["label"]
        self.chatbot.append(["生成报告", f"正在整合{domain_label}递进式精读结果，生成深度技术报告..."])
        yield from update_ui(chatbot=self.chatbot, history=self.history)

        # 统一构建报告提示（避免重复代码）
        prompt_prefix = {
            "rf_ic": (
                "请将以下对RF IC论文的递进式精读分析整理为完整的技术报告。"
                "报告应体现RF IC专业分析的递进逻辑：从问题域理解→理论构建→技术实现→实验验证→批判分析→工程复现→架构可视化→影响评估→要点总结。"
                "同时包含RF IC专业分析：电路架构→工艺技术→性能指标→设计方法→制造测试→应用市场。"
                "层次清晰，突出RF IC设计的技术深度和工程价值，不包含任何代码/伪代码/命令行。"
                "若包含```mermaid 代码块，请原样保留。\n\n"
            ),
            "general": (
                "请将以下递进式精读分析整理为完整的技术报告。"
                "报告应体现分析的递进逻辑：从问题域理解→理论构建→技术实现→实验验证→批判分析→工程复现→架构可视化→影响评估→要点总结。"
                "层次清晰，突出核心思想与实验设计要点，不包含任何代码/伪代码/命令行。"
                "若包含```mermaid 代码块，请原样保留。\n\n"
            )
        }
        
        prompt = prompt_prefix.get(self.paper_domain, prompt_prefix["general"]) + self._get_domain_config()["result_label"]
        
        # 按照递进顺序组织分析结果
        layer_order = [
            "problem_domain_and_motivation",
            "theoretical_framework_and_contributions", 
            "method_design_and_technical_details",
            "experimental_validation_and_effectiveness",
            "worth_reading_judgment",
            # RF IC特有
            "rf_ic_circuit_architecture_detail",
            "rf_ic_performance_and_methods",
            "rf_ic_manufacturing_market_analysis",
            # 演示材料
            "presentation_summary_and_materials",
        ]
        
        # 如果是RF IC论文，添加RF IC专用问题（已在layer_order中）
        if self.paper_domain == "rf_ic":
            rf_ic_layer_order = []
            layer_order.extend(rf_ic_layer_order)
        
        for layer_id in layer_order:
            for q in self.questions:
                if q.id == layer_id and q.id in self.results:
                    prompt += f"\n\n## {q.description}\n{self.results[q.id]}"
                    break

        if self.paper_domain == "rf_ic":
            sys_prompt = (
                "以递进式RF IC深度分析为主线组织报告：体现从宏观到微观、从理论到实践的完整分析链条，"
                "同时突出RF IC专业特色。每个部分都要与前面的分析形成逻辑关联，突出递进关系。"
                "以RF IC工程实现为目标，背景极简，方法与实现细节充分，条理分明，包含必要的清单与步骤。"
                "强调技术深度、工程价值和产业化前景。"
            )
        else:
            sys_prompt = (
                "以递进式深度分析为主线组织报告：体现从宏观到微观、从理论到实践的完整分析链条。"
                "每个部分都要与前面的分析形成逻辑关联，突出递进关系。"
                "以工程复现为目标，背景极简，方法与实现细节充分，条理分明，包含必要的清单与步骤。"
            )

        resp = yield from request_gpt_model_in_new_thread_with_ui_alive(
            inputs=prompt,
            inputs_show_user=f"生成{domain_label}递进式论文精读技术报告",
            llm_kwargs=self.llm_kwargs,
            chatbot=self.chatbot,
            history=[],
            sys_prompt=sys_prompt,
        )
        
        # 记录报告生成的token使用
        if resp:
            try:
                self._token_inputs.append(prompt)
                self._token_outputs.append(resp)
            except Exception:
                pass
        
        return resp or "报告生成失败"

    def _extract_secondary_category(self, report: str) -> str:
        """从报告中提取"归属："后的二级分类文本，只保留类似
        "7. 机器学习辅助设计 (ML-Aided RF Design) -> 系统级建模与快速综合"。
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
        domain_cfg = self._get_domain_config()
        domain_prefix = domain_cfg["file_prefix"]
        pdf_basename = f"未知{domain_prefix}论文"
        if self.paper_file_path and os.path.exists(self.paper_file_path):
            pdf_basename = os.path.splitext(os.path.basename(self.paper_file_path))[0]
            pdf_basename = re.sub(r'[^\w\u4e00-\u9fff]', '_', pdf_basename)
            if len(pdf_basename) > 50:
                pdf_basename = pdf_basename[:50]

        parts: List[str] = []
        domain_title = domain_cfg["report_title"]
        parts.append(f"{domain_title}\n\n{report}")
        
        # 优先追加执行级摘要、流程图与PPT材料
        #（已移除独立执行摘要与独立流程图追加）
        if "presentation_summary_and_materials" in self.results:
            parts.append(f"\n\n## 演示材料\n\n{self.results['presentation_summary_and_materials']}")
         
        # 追加其余维度
        for q in self.questions:
            if q.id in self.results and q.id not in {"presentation_summary_and_materials"}:
                parts.append(f"\n\n## {q.description}\n\n{self.results[q.id]}")

        # 追加 Token 估算结果
        try:
            # 使用新的token统计方法
            token_summary = self._get_token_usage_summary()
            if token_summary and "Token统计信息获取失败" not in token_summary:
                parts.append(f"\n\n{token_summary}")
            
            # 保留原有的估算方法作为补充
            stats = estimate_token_usage(self._token_inputs, self._token_outputs, self.llm_kwargs.get('llm_model', 'gpt-3.5-turbo'))
            if stats and stats.get('sum_total_tokens', 0) > 0:
                parts.append(
                    "\n\n## 详细Token估算\n\n"
                    f"- 模型: {stats.get('model')}\n"
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
            file_basename=f"{timestamp}_{pdf_basename}_{domain_prefix}精读报告.md",
        )
        if result_file and os.path.exists(result_file):
            promote_file_to_downloadzone(result_file, chatbot=self.chatbot)
            return result_file
        return None

    def analyze_paper(self, paper_path: str) -> Generator:
        # 加载论文
        ok = yield from self._load_paper(paper_path)
        if not ok:
            return None
        
        # 主题分类判断
        yield from self._classify_paper_domain()
        
        # 根据领域获取相应的问题列表
        domain_questions = self._get_domain_specific_questions()
        
        # 分析关键问题
        for q in domain_questions:
            yield from self._ask(q)
        
        # 生成总结报告
        report = yield from self._generate_report()
        
        # 从报告中提取二级分类归属
        self.secondary_category = self._extract_secondary_category(report)
        
        # 生成 YAML 头
        self.yaml_header = yield from self._generate_yaml_header()
        
        # 保存报告
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
    """主函数 - 批量论文精读（支持领域自适应）"""
    chatbot.append([
        "函数插件功能及使用方式",
        (
            "批量论文精读：智能识别论文主题（通用/RF IC），自动选择最适合的递进式精读分析策略，"
            "为每篇论文生成面向实现与复现的深度技术报告。\n\n"
            "使用方式：\n1) 输入包含多个PDF的文件夹路径；\n2) 或输入多个论文ID（DOI或arXiv），用逗号分隔；\n3) 点击开始。\n\n"
            "智能分析特性：\n- 自动主题分类（通用论文 vs RF IC论文）\n- 递进式深度分析（9层递进逻辑）\n- 领域特定专业问题\n- 统一的报告格式和YAML元信息\n\n"
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

    chatbot.append(["开始智能批量精读", f"找到 {len(paper_files)} 篇论文，开始智能主题分类和递进式深度分析..."])
    yield from update_ui(chatbot=chatbot, history=history)

    analyzer = BatchPaperDetailAnalyzer(llm_kwargs, plugin_kwargs, chatbot, history, system_prompt)

    successes: List[str] = []
    failures: List[str] = []
    domain_stats = {"general": 0, "rf_ic": 0}

    for i, paper_file in enumerate(paper_files):
        try:
            chatbot.append([f"精读论文 {i+1}/{len(paper_files)}", f"正在智能精读: {os.path.basename(paper_file)}"])
            yield from update_ui(chatbot=chatbot, history=history)
            outfile = yield from analyzer.analyze_paper(paper_file)
            if outfile:
                successes.append((os.path.basename(paper_file), outfile, analyzer.paper_domain))
                domain_stats[analyzer.paper_domain] += 1
                domain_label = "RF IC" if analyzer.paper_domain == "rf_ic" else "通用"
                chatbot.append([f"完成论文 {i+1}/{len(paper_files)}", f"成功生成{domain_label}精读报告: {os.path.basename(outfile)}"])
            else:
                failures.append(os.path.basename(paper_file))
                chatbot.append([f"失败论文 {i+1}/{len(paper_files)}", f"分析失败: {os.path.basename(paper_file)}"])
            yield from update_ui(chatbot=chatbot, history=history)
        except Exception as e:
            failures.append(os.path.basename(paper_file))
            chatbot.append([f"错误论文 {i+1}/{len(paper_files)}", f"分析出错: {os.path.basename(paper_file)} - {str(e)}"])
            yield from update_ui(chatbot=chatbot, history=history)

    summary = "智能批量精读完成！\n\n"
    summary += "📊 分析统计：\n"
    summary += f"- 总论文数：{len(paper_files)}\n"
    summary += f"- 成功分析：{len(successes)}\n"
    summary += f"- 分析失败：{len(failures)}\n\n"
    
    summary += "🎯 主题分类统计：\n"
    summary += f"- 通用论文：{domain_stats['general']} 篇\n"
    summary += f"- RF IC论文：{domain_stats['rf_ic']} 篇\n\n"
    
    if successes:
        summary += "✅ 成功生成报告：\n"
        for paper_name, report_path, domain in successes:
            domain_label = "RF IC" if domain == "rf_ic" else "通用"
            summary += f"- {paper_name} ({domain_label}) → {os.path.basename(report_path)}\n"
    if failures:
        summary += "\n❌ 分析失败的论文：\n"
        for name in failures:
            summary += f"- {name}\n"

    chatbot.append(["智能批量精读完成", summary])
    yield from update_ui(chatbot=chatbot, history=history)


