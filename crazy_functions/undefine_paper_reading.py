import json
import re
import os
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Generator, Tuple, Optional
from crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive
from toolbox import update_ui, promote_file_to_downloadzone, write_history_to_file, CatchException, report_exception
from shared_utils.fastapi_server import validate_path_safety
from crazy_functions.paper_fns.paper_download import extract_paper_id, get_arxiv_paper, format_arxiv_id
import difflib


def _estimate_tokens(text: str, llm_model: str) -> int:
    """使用已配置模型的tokenizer估算文本token数。"""
    try:
        from request_llms.bridge_all import model_info
        cnt_fn = model_info.get(llm_model, {}).get("token_cnt", None)
        if cnt_fn is None:
            # 兜底：若模型未配置，使用gpt-3.5的tokenizer近似
            cnt_fn = model_info["gpt-3.5-turbo"]["token_cnt"]
        return int(cnt_fn(text or ""))
    except Exception:
        # 无法估计时以字符数近似
        return len(text or "")


def estimate_token_usage(inputs: List[str], outputs: List[str], llm_model: str) -> Dict:
    """
    独立的检测函数：估算一组交互的输入/输出token消耗。
    """
    n = max(len(inputs or []), len(outputs or []))
    items = []
    sum_in = 0
    sum_out = 0
    for i in range(n):
        inp = inputs[i] if i < len(inputs) else ""
        out = outputs[i] if i < len(outputs) else ""
        ti = _estimate_tokens(inp, llm_model)
        to = _estimate_tokens(out, llm_model)
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
class PaperQuestion:
    """论文分析问题类"""
    id: str  # 问题ID
    question: str  # 问题内容
    importance: int  # 重要性 (1-5，5最高)
    description: str  # 问题描述
    domain: str  # 适用领域 ("general", "rf_ic", "both")


class UnifiedBatchPaperAnalyzer:
    """统一的批量论文分析器 - 支持主题分类和动态prompt"""

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
        self.secondary_category = None
        self.paper_domain = "general"  # 论文领域分类
        self.context_history = []  # 与LLM共享的上下文（每篇论文注入一次全文）
        # 统计用：记录每次LLM交互的输入与输出
        self._token_inputs: List[str] = []
        self._token_outputs: List[str] = []
        
        # ---------- 读取分类树 ----------
        json_path = os.path.join(os.path.dirname(__file__), 'paper.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            self.category_tree = json.load(f)          # Dict[str, List[str]]

        # 生成给 LLM 的当前分类清单
        category_lines = [f"{main} -> {', '.join(subs)}"
                        for main, subs in self.category_tree.items()]
        self.category_prompt_str = '\n'.join(category_lines)

        # 定义速读问题库（精简版，专注于快速筛选）
        self.questions = [
            # 通用速读问题（适用于所有论文）
            PaperQuestion(
                id="research_methods_and_data",
                question="请简要概括论文的核心内容：1) 研究问题是什么？2) 主要方法/技术路线是什么？3) 实验数据来源如何？",
                importance=5,
                description="研究问题与方法概述",
                domain="both"
            ),
            PaperQuestion(
                id="findings_innovations_and_impact",
                question="请总结论文的主要发现与创新：1) 核心结果是什么？2) 主要创新点有哪些？3) 对领域的影响如何？",
                importance=4,
                description="主要发现与创新点",
                domain="both"
            ),
            PaperQuestion(
                id="ppt_md_summary",
                question=(
                    "请输出用于PPT的Markdown极简摘要（仅按如下结构，勿嵌入代码块）：\n\n"
                    "# 总述（1 行）\n"
                    "- 用一句话概括论文做了什么、为何有效\n\n"
                    "# 核心要点（3-5条）\n"
                    "- 关键输入/方法/输出/创新（每条 ≤ 16 字）\n\n"
                    "# 应用与效果（≤ 3 条，可省略）\n"
                    "- 场景/指标/收益"
                ),
                importance=3,
                description="PPT 用极简Markdown摘要",
                domain="both"
            ),
            PaperQuestion(
                id="worth_reading_judgment",
                question="请综合评估这篇论文是否值得精读，并给出明确的推荐等级：\n1) **创新性**：是否具有开创性贡献？\n2) **可靠性**：研究方法是否严谨？\n3) **影响力**：是否可能产生重要影响？\n4) **综合建议**：给出\"强烈推荐\"、\"推荐\"、\"一般\"或\"不推荐\"的评级，并简要说明理由。",
                importance=5,
                description="是否值得精读",
                domain="both"
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
                description="论文二级分类归属",
                domain="both"
            ),
            
            # RF IC专用速读问题（简化版）
            PaperQuestion(
                id="rf_ic_design_and_metrics",
                question="请简要分析RF IC论文的技术要点：1) 电路架构特点是什么？2) 主要性能指标如何？3) 设计创新点在哪里？",
                importance=4,
                description="RF IC技术要点概述",
                domain="rf_ic"
            ),
            PaperQuestion(
                id="rf_ic_applications_challenges_future",
                question="请评估RF IC论文的应用价值：1) 目标应用场景是什么？2) 技术难点在哪里？3) 产业化前景如何？",
                importance=3,
                description="RF IC应用与前景评估",
                domain="rf_ic"
            ),
            PaperQuestion(
                id="rf_ic_ppt_md_summary",
                question=(
                    "请输出用于PPT的RF IC方向Markdown极简摘要（仅按如下结构，勿嵌入代码块）：\n\n"
                    "# 总述（1 行）\n"
                    "- 用一句话概括该电路/系统做了什么、为何有效\n\n"
                    "# 电路/设计要点（3-5条）\n"
                    "- 核心模块/信号流/关键设计\n\n"
                    "# 性能与应用\n"
                    "- 指标/场景/收益"
                ),
                importance=3,
                description="RF IC PPT 用极简Markdown摘要",
                domain="rf_ic"
            ),
        ]

        # 按重要性排序
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
                    self.chatbot.append(["主题分类", "检测到RF IC相关论文，将使用专业RF IC分析策略"])
                else:
                    self.paper_domain = "general"
                    self.chatbot.append(["主题分类", "检测到通用论文，将使用通用分析策略"])
            else:
                self.paper_domain = "general"
                self.chatbot.append(["主题分类", "无法确定主题，使用通用分析策略"])

            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return True

        except Exception as e:
            self.paper_domain = "general"
            self.chatbot.append(["分类错误", f"主题分类失败，使用通用策略: {str(e)}"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return False

    def _get_domain_specific_questions(self) -> List[PaperQuestion]:
        """根据论文领域获取相应的问题列表"""
        if self.paper_domain == "rf_ic":
            # RF IC论文：包含RF IC专用问题和核心通用问题
            return [q for q in self.questions if q.domain in ["both", "rf_ic"]]
        else:
            # 通用论文：只包含通用问题
            return [q for q in self.questions if q.domain in ["both", "general"]]

    def _get_domain_specific_system_prompt(self) -> str:
        """根据论文领域获取相应的系统提示"""
        if self.paper_domain == "rf_ic":
            return """你是一个专业的射频集成电路(RF IC)分析专家，具有深厚的电路设计、半导体工艺和无线通信系统知识。请从RF IC专业角度深入分析论文，使用准确的术语，提供有见地的技术评估。"""
        else:
            return """你是一个专业的科研论文分析助手，需要仔细阅读论文内容并回答问题。请保持客观、准确，并基于论文内容提供深入分析。"""

    def _get_domain_specific_analysis_prompt(self, question: PaperQuestion) -> str:
        """根据论文领域和问题生成相应的分析提示"""
        if self.paper_domain == "rf_ic":
            return f"""请基于已记住的射频集成电路论文全文，从RF IC专业角度简要回答：

问题：{question.question}

请保持简洁明了，重点关注技术创新点和应用价值。"""
        else:
            return f"请基于已记住的论文全文简要回答：{question.question}"

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
        with open(path, 'w', encoding='utf-8') as f:
            for kw in sorted(set(keywords), key=lambda x: x.lower()):
                f.write(kw + '\n')

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
                "doi: [DOI链接, None]\n"
                "journal_or_conference: [期刊或会议名称, None]\n"
                "year: [年份, None]\n"
                "source_code: [源码链接, None]\n"
                "read_status: [已阅读, 未阅读]\n"
                "仅输出以 --- 开始、以 --- 结束的 YAML Front Matter，不要附加其他文本。read_status默认未阅读。"
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
                    raw_list = [x.strip().strip('\"\'\'') for x in inner.split(',') if x.strip()]
                    merged, _ = self._merge_keywords_with_db(raw_list)
                    # 以原样式写回（使用引号包裹，避免 YAML 解析问题）
                    rebuilt = ', '.join([f'\"{k}\"' for k in merged])
                    text = re.sub(r"^keywords:\s*\[(.*?)\]\s*$", f"keywords: [{rebuilt}]", text, flags=re.MULTILINE)
                
                # 注入"归属"二级分类（若可用）
                try:
                    if getattr(self, 'secondary_category', None):
                        escaped = self.secondary_category.replace('\"', '\\\"')
                        if text.endswith("---"):
                            text = text[:-3].rstrip() + f"\nsecondary_category: \"{escaped}\"\n---"
                except Exception:
                    pass
                
                # 简化星级评分映射（速读版）：先移除已有 stars，再按评级追加一次
                try:
                    judge = self.results.get("worth_reading_judgment", "")
                    stars = "⭐⭐⭐"  # 默认
                    if isinstance(judge, str) and judge:
                        if "强烈推荐" in judge:
                            stars = "⭐⭐⭐⭐⭐"
                        elif "推荐" in judge:
                            stars = "⭐⭐⭐⭐"
                        elif "谨慎" in judge:
                            stars = "⭐⭐"
                        elif "不推荐" in judge:
                            stars = "⭐"

                    # 移除原有的 stars 行（标量或列表形式）
                    text = re.sub(r"^stars:\s*\[.*?\]\s*$\n?", "", text, flags=re.MULTILINE)
                    text = re.sub(r"^stars:\s*.*$\n?", "", text, flags=re.MULTILINE)

                    # 统一仅追加一次列表形式的 stars 字段
                    if text.endswith("---"):
                        text = text[:-3].rstrip() + f"\nstars: [\"{stars}\"]\n---"
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

    def _analyze_question(self, question: PaperQuestion) -> Generator:
        """分析单个问题 - 根据领域动态调整分析策略"""
        try:
            # 根据论文领域生成相应的分析提示
            prompt = self._get_domain_specific_analysis_prompt(question)
            
            # 获取领域特定的系统提示
            sys_prompt = self._get_domain_specific_system_prompt()

            response = yield from request_gpt_model_in_new_thread_with_ui_alive(
                inputs=prompt,
                inputs_show_user=question.question,
                llm_kwargs=self.llm_kwargs,
                chatbot=self.chatbot,
                history=self.context_history or [],
                sys_prompt=sys_prompt
            )

            if response:
                self.results[question.id] = response
                # 记录本轮交互的输入与输出用于token估算
                self._token_inputs.append(prompt)
                self._token_outputs.append(response)

                # 如果是分类归属问题，自动更新 paper.json
                if question.id == "category_assignment":
                    self._update_category_json(response)
                    try:
                        mcat = re.search(r"^归属：\s*([^\r\n]+)", response, flags=re.MULTILINE)
                        if mcat:
                            self.secondary_category = mcat.group(1).strip()
                    except Exception:
                        pass

                return True
            return False

        except Exception as e:
            self.chatbot.append(["错误", f"分析问题时出错: {str(e)}"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return False

    def _generate_summary(self) -> Generator:
        """生成速读筛选报告"""
        domain_label = "RF IC" if self.paper_domain == "rf_ic" else "通用"
        self.chatbot.append(["生成速读报告", f"正在整合{domain_label}论文速读分析结果，生成筛选报告..."])
        yield from update_ui(chatbot=self.chatbot, history=self.history)

        if self.paper_domain == "rf_ic":
            summary_prompt = """请基于以下对RF IC论文的速读分析，生成一份简洁的论文筛选报告。

报告要求：
1. 简明扼要地总结论文的核心技术要点
2. 突出RF IC设计的创新点和价值
3. 评估技术的应用前景和成熟度
4. 明确给出是否值得精读的建议及理由

请保持简洁明了，适合快速决策。"""
        else:
            summary_prompt = """请基于以下对论文的速读分析，生成一份简洁的论文筛选报告。

报告要求：
1. 简明扼要地总结论文的核心内容
2. 突出研究的主要创新点和贡献
3. 评估研究的价值和影响
4. 明确给出是否值得精读的建议及理由

请保持简洁明了，适合快速决策。"""

        for q in self.questions:
            if q.id in self.results:
                summary_prompt += f"\n\n{q.description}:\n{self.results[q.id]}"

        try:
            # 使用单线程版本的请求函数，可以在前端实时显示生成结果
            response = yield from request_gpt_model_in_new_thread_with_ui_alive(
                inputs=summary_prompt,
                inputs_show_user=f"生成{domain_label}论文速读筛选报告",
                llm_kwargs=self.llm_kwargs,
                chatbot=self.chatbot,
                history=[],
                sys_prompt=f"你是一个{'射频集成电路领域的专家' if self.paper_domain == 'rf_ic' else '科研论文评审专家'}，请将速读分析整合为一份简洁的筛选报告。报告应当重点突出论文的核心价值和创新点，并明确给出是否值得精读的建议。保持简洁明了，便于快速决策。"
            )

            if response:
                # 记录报告生成的token使用
                self._token_inputs.append(summary_prompt)
                self._token_outputs.append(response)
                return response
            return "速读报告生成失败"

        except Exception as e:
            self.chatbot.append(["错误", f"生成速读报告时出错: {str(e)}"])
            yield from update_ui(chatbot=self.chatbot, history=self.history)
            return "速读报告生成失败: " + str(e)

    def save_report(self, report: str, paper_file_path: str = None) -> str:
        """保存分析报告，返回保存的文件路径"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 获取PDF文件名（不含扩展名）
        domain_prefix = "RF_IC" if self.paper_domain == "rf_ic" else "通用"
        pdf_filename = f"未知{domain_prefix}论文"
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
            domain_title = "射频集成电路论文速读筛选报告" if self.paper_domain == "rf_ic" else "学术论文速读筛选报告"
            md_parts.append(f"{domain_title}\n\n{report}")

            # 速读报告：简洁组织内容，重点关注筛选决策
            if self.paper_domain == "rf_ic":
                # RF IC论文速读：按重要性组织
                # 1. 核心分析
                core_questions = ["research_methods_and_data", "findings_innovations_and_impact", "rf_ic_design_and_metrics", "rf_ic_applications_challenges_future"]
                for q_id in core_questions:
                    for q in self.questions:
                        if q.id == q_id and q.id in self.results:
                            md_parts.append(f"\n\n## 📋 {q.description}\n\n{self.results[q.id]}")
                            break
                
                # 2. 阅读建议（最重要）
                if "worth_reading_judgment" in self.results:
                    md_parts.append(f"\n\n## 🎯 是否值得精读\n\n{self.results['worth_reading_judgment']}")
                
                # 3. 分类信息
                if "category_assignment" in self.results:
                    md_parts.append(f"\n\n## 📂 论文分类\n\n{self.results['category_assignment']}")

                # 4. PPT 摘要
                if "rf_ic_ppt_md_summary" in self.results:
                    md_parts.append(f"\n\n## 📝 RF IC PPT 摘要\n\n{self.results['rf_ic_ppt_md_summary']}")
            else:
                # 通用论文速读：按重要性组织
                # 1. 核心分析
                core_questions = ["research_methods_and_data", "findings_innovations_and_impact"]
                for q_id in core_questions:
                    for q in self.questions:
                        if q.id == q_id and q.id in self.results:
                            md_parts.append(f"\n\n## 📋 {q.description}\n\n{self.results[q.id]}")
                            break
                
                # 2. 阅读建议（最重要）
                if "worth_reading_judgment" in self.results:
                    md_parts.append(f"\n\n## 🎯 是否值得精读\n\n{self.results['worth_reading_judgment']}")
                
                # 3. 分类信息
                if "category_assignment" in self.results:
                    md_parts.append(f"\n\n## 📂 论文分类\n\n{self.results['category_assignment']}")

                # 4. PPT 摘要
                if "ppt_md_summary" in self.results:
                    md_parts.append(f"\n\n## 📝 PPT 摘要\n\n{self.results['ppt_md_summary']}")

            md_content = "".join(md_parts)

            # 若已生成 YAML 头，则置于文首
            if hasattr(self, 'yaml_header') and self.yaml_header:
                md_content = f"{self.yaml_header}\n\n" + md_content

            # 追加简化的分析统计
            try:
                stats = estimate_token_usage(self._token_inputs, self._token_outputs, self.llm_kwargs.get('llm_model', 'gpt-3.5-turbo'))
                if stats and stats.get('sum_total_tokens', 0) > 0:
                    md_content += (
                        "\n\n## 📊 分析统计\n\n"
                        f"- 分析模型: {stats.get('model')}\n"
                        f"- Token消耗: {stats.get('sum_total_tokens', 0)} tokens\n"
                    )
            except Exception:
                pass

            result_file = write_history_to_file(
                history=[md_content],
                file_basename=f"{timestamp}_{pdf_filename}_{domain_prefix}解读报告.md"
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
        # 每篇论文独立统计 token：重置交互记录
        self._token_inputs = []
        self._token_outputs = []
        # 加载论文
        success = yield from self._load_paper(paper_path)
        if not success:
            return None

        # 主题分类判断
        yield from self._classify_paper_domain()

        # 根据领域获取相应的问题列表
        domain_questions = self._get_domain_specific_questions()

        # 分析关键问题
        for question in domain_questions:
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
    base_save_dir = get_log_folder(get_user(chatbot), plugin_name='unified_paper_download')
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
def 统一批量论文速读(txt: str, llm_kwargs: Dict, plugin_kwargs: Dict, chatbot: List,
             history: List, system_prompt: str, user_request: str):
    """主函数 - 统一批量论文速读（支持主题分类）"""
    # 初始化分析器
    chatbot.append(["函数插件功能及使用方式", "统一批量论文速读：快速筛选论文，判断是否值得精读。智能识别论文主题（通用/RF IC），为每篇论文生成简洁的速读报告。 <br><br>📋 使用方式：<br>1、输入包含多个PDF文件的文件夹路径<br>2、或者输入多个论文ID（DOI或arXiv ID），用逗号分隔<br>3、点击插件开始快速筛选分析<br><br>🎯 速读特性：<br>- 快速识别论文核心内容和创新点<br>- 自动主题分类（通用论文 vs RF IC论文）<br>- 明确给出是否值得精读的建议<br>- 简洁的报告格式，便于快速决策"])
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

    chatbot.append(["开始智能批量分析", f"找到 {len(paper_files)} 篇论文，开始智能主题分类和批量分析..."])
    yield from update_ui(chatbot=chatbot, history=history)

    # 创建统一分析器
    analyzer = UnifiedBatchPaperAnalyzer(llm_kwargs, plugin_kwargs, chatbot, history, system_prompt)
    
    # 批量分析每篇论文
    successful_reports = []
    failed_papers = []
    domain_stats = {"general": 0, "rf_ic": 0}
    
    for i, paper_file in enumerate(paper_files):
        try:
            chatbot.append([f"分析论文 {i+1}/{len(paper_files)}", f"正在智能分析: {os.path.basename(paper_file)}"])
            yield from update_ui(chatbot=chatbot, history=history)
            
            # 分析单篇论文
            saved_file = yield from analyzer.analyze_paper(paper_file)
            
            if saved_file:
                successful_reports.append((os.path.basename(paper_file), saved_file, analyzer.paper_domain))
                domain_stats[analyzer.paper_domain] += 1
                chatbot.append([f"完成论文 {i+1}/{len(paper_files)}", f"成功分析并保存报告: {os.path.basename(saved_file)} (领域: {analyzer.paper_domain})"])
            else:
                failed_papers.append(os.path.basename(paper_file))
                chatbot.append([f"失败论文 {i+1}/{len(paper_files)}", f"分析失败: {os.path.basename(paper_file)}"])
            
            yield from update_ui(chatbot=chatbot, history=history)
            
        except Exception as e:
            failed_papers.append(os.path.basename(paper_file))
            chatbot.append([f"错误论文 {i+1}/{len(paper_files)}", f"分析出错: {os.path.basename(paper_file)} - {str(e)}"])
            yield from update_ui(chatbot=chatbot, history=history)

    # 生成批量分析总结
    summary = f"智能批量分析完成！\n\n"
    summary += f"📊 分析统计：\n"
    summary += f"- 总论文数：{len(paper_files)}\n"
    summary += f"- 成功分析：{len(successful_reports)}\n"
    summary += f"- 分析失败：{len(failed_papers)}\n\n"
    
    summary += f"🎯 主题分类统计：\n"
    summary += f"- 通用论文：{domain_stats['general']} 篇\n"
    summary += f"- RF IC论文：{domain_stats['rf_ic']} 篇\n\n"
    
    if successful_reports:
        summary += f"✅ 成功生成报告：\n"
        for paper_name, report_path, domain in successful_reports:
            domain_label = "RF IC" if domain == "rf_ic" else "通用"
            summary += f"- {paper_name} ({domain_label}) → {os.path.basename(report_path)}\n"
    
    if failed_papers:
        summary += f"\n❌ 分析失败的论文：\n"
        for paper_name in failed_papers:
            summary += f"- {paper_name}\n"

    chatbot.append(["智能批量分析完成", summary])
    yield from update_ui(chatbot=chatbot, history=history)
