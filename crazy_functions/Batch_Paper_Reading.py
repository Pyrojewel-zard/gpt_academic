import os
import time
import glob
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Generator, Tuple
from crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive
from toolbox import update_ui, promote_file_to_downloadzone, write_history_to_file, CatchException, report_exception
from shared_utils.fastapi_server import validate_path_safety
from crazy_functions.paper_fns.paper_download import extract_paper_id, extract_paper_ids, get_arxiv_paper, format_arxiv_id

# 新增：请求限流和重试控制
import queue
import random
from threading import Semaphore 


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
        self.progress_lock = threading.Lock()  # 进度锁
        self.current_progress = 0
        self.total_papers = 0
        
        # 新增：详细进度跟踪
        self.current_paper_name = ""
        self.current_question_index = 0
        self.total_questions = 0
        self.processing_stage = ""  # 当前处理阶段：loading, analyzing, summarizing, saving
        
        # 新增：请求限流控制
        self.request_semaphore = Semaphore(2)  # 最多同时2个API请求
        self.retry_delay = 5  # 重试延迟（秒）
        self.max_retries = 3  # 最大重试次数

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
        ]

        # 按重要性排序
        self.questions.sort(key=lambda q: q.importance, reverse=True)
        self.total_questions = len(self.questions)

    def update_detailed_progress(self, paper_name: str, stage: str, question_index: int = -1, status: str = ""):
        """更新详细处理进度"""
        with self.progress_lock:
            self.current_paper_name = paper_name
            self.processing_stage = stage
            if question_index >= 0:
                self.current_question_index = question_index
            
            # 构建进度消息
            progress_msg = f"📊 批量论文分析进度\n\n"
            progress_msg += f"📄 当前论文: {paper_name}\n"
            progress_msg += f"🔄 处理阶段: {stage}\n"
            
            if stage == "analyzing" and question_index >= 0:
                question = self.questions[question_index]
                progress_msg += f"❓ 当前问题 ({question_index + 1}/{self.total_questions}): {question.description}\n"
                progress_msg += f"📈 问题进度: {question_index + 1}/{self.total_questions}\n"
            
            if status:
                progress_msg += f"📋 状态: {status}\n"
            
            # 更新进度显示
            if len(self.chatbot) > 0 and "批量论文分析进度" in self.chatbot[-1][0]:
                self.chatbot[-1] = ["批量论文分析进度", progress_msg]
            else:
                self.chatbot.append(["批量论文分析进度", progress_msg])

    def update_paper_progress(self, paper_name: str, status: str):
        """更新论文处理进度（简化版本，用于最终状态）"""
        with self.progress_lock:
            self.current_progress += 1
            progress_percent = (self.current_progress / self.total_papers) * 100
            progress_msg = f"📊 论文处理进度: {self.current_progress}/{self.total_papers} ({progress_percent:.1f}%) - {paper_name}: {status}"
            
            # 更新进度显示
            if len(self.chatbot) > 0 and "论文处理进度" in self.chatbot[-1][0]:
                self.chatbot[-1] = ["论文处理进度", progress_msg]
            else:
                self.chatbot.append(["论文处理进度", progress_msg])

    def _load_paper(self, paper_path: str) -> bool:
        """加载论文内容 - 非生成器版本，用于多线程"""
        from crazy_functions.doc_fns.text_content_loader import TextContentLoader
        
        # 保存论文文件路径
        self.paper_file_path = paper_path
        paper_name = os.path.basename(paper_path)
        
        # 更新进度
        self.update_detailed_progress(paper_name, "loading", status="正在加载论文内容...")

        try:
            # 使用TextContentLoader读取文件
            loader = TextContentLoader(self.chatbot, self.history)
            
            # 执行文件加载
            for _ in loader.execute_single_file(paper_path):
                pass  # 忽略生成器输出
            
            # 获取加载的内容
            if len(self.history) >= 2 and self.history[-2]:
                self.paper_content = self.history[-2]
                self.update_detailed_progress(paper_name, "loading", status="✅ 论文加载完成")
                return True
            else:
                self.update_detailed_progress(paper_name, "loading", status="❌ 论文加载失败")
                return False
        except Exception as e:
            self.update_detailed_progress(paper_name, "loading", status=f"❌ 加载错误: {str(e)}")
            return False

    def _analyze_question(self, question: PaperQuestion, question_index: int) -> bool:
        """分析单个问题 - 非生成器版本，用于多线程"""
        paper_name = os.path.basename(self.paper_file_path) if self.paper_file_path else "未知论文"
        
        # 更新进度
        self.update_detailed_progress(paper_name, "analyzing", question_index, f"正在分析: {question.description}")
        
        # 使用信号量限制并发请求
        with self.request_semaphore:
            for retry_count in range(self.max_retries):
                try:
                    # 添加随机延迟，避免请求冲突
                    if retry_count > 0:
                        delay = self.retry_delay + random.uniform(1, 3)
                        time.sleep(delay)
                        self.update_detailed_progress(paper_name, "analyzing", question_index, 
                                                   f"重试分析 ({retry_count + 1}/{self.max_retries}): {question.description}")
                    
                    # 检查论文内容长度，避免超出API限制
                    content_length = len(self.paper_content)
                    if content_length > 80000:  # 设置安全阈值
                        # 截取论文内容的核心部分
                        truncated_content = self.paper_content[:80000] + "\n\n[内容已截取，保留核心部分]"
                        self.update_detailed_progress(paper_name, "analyzing", question_index, 
                                                   f"论文内容过长({content_length}字符)，已截取至80000字符")
                    else:
                        truncated_content = self.paper_content
                    
                    # 创建分析提示
                    prompt = f"请基于以下论文内容回答问题：\n\n{truncated_content}\n\n问题：{question.question}"

                    # 使用单线程版本的请求函数
                    response = None
                    for resp in request_gpt_model_in_new_thread_with_ui_alive(
                        inputs=prompt,
                        inputs_show_user=question.question,  # 显示问题本身
                        llm_kwargs=self.llm_kwargs,
                        chatbot=self.chatbot,
                        history=[],  # 空历史，确保每个问题独立分析
                        sys_prompt="你是一个专业的科研论文分析助手，需要仔细阅读论文内容并回答问题。请保持客观、准确，并基于论文内容提供深入分析。"
                    ):
                        response = resp

                    if response:
                        self.results[question.id] = response
                        self.update_detailed_progress(paper_name, "analyzing", question_index, f"✅ 完成: {question.description}")
                        return True
                    else:
                        if retry_count < self.max_retries - 1:
                            self.update_detailed_progress(paper_name, "analyzing", question_index, 
                                                       f"⚠️ 分析失败，准备重试: {question.description}")
                            continue
                        else:
                            self.update_detailed_progress(paper_name, "analyzing", question_index, f"❌ 最终失败: {question.description}")
                            return False

                except Exception as e:
                    error_msg = str(e)
                    if retry_count < self.max_retries - 1:
                        self.update_detailed_progress(paper_name, "analyzing", question_index, 
                                                   f"⚠️ 分析出错，准备重试: {question.description} - 错误: {error_msg}")
                        continue
                    else:
                        self.update_detailed_progress(paper_name, "analyzing", question_index, 
                                                   f"❌ 最终错误: {question.description} - {error_msg}")
                        return False
            
            return False

    def _generate_summary(self) -> str:
        """生成最终总结报告 - 非生成器版本，用于多线程"""
        paper_name = os.path.basename(self.paper_file_path) if self.paper_file_path else "未知论文"
        
        # 更新进度
        self.update_detailed_progress(paper_name, "summarizing", status="正在生成总结报告...")
        
        # 使用信号量限制并发请求
        with self.request_semaphore:
            for retry_count in range(self.max_retries):
                try:
                    # 添加随机延迟，避免请求冲突
                    if retry_count > 0:
                        delay = self.retry_delay + random.uniform(1, 3)
                        time.sleep(delay)
                        self.update_detailed_progress(paper_name, "summarizing", 
                                                   status=f"重试生成总结报告 ({retry_count + 1}/{self.max_retries})...")
                    
                    summary_prompt = "请基于以下对论文的各个方面的分析，生成一份全面的论文解读报告。报告应该简明扼要地呈现论文的关键内容，并保持逻辑连贯性。"

                    for q in self.questions:
                        if q.id in self.results:
                            summary_prompt += f"\n\n关于{q.description}的分析:\n{self.results[q.id]}"
                    
                    # 检查提示长度，避免超出API限制
                    if len(summary_prompt) > 80000:
                        # 截取提示内容
                        summary_prompt = summary_prompt[:80000] + "\n\n[提示内容已截取，保留核心部分]"
                        self.update_detailed_progress(paper_name, "summarizing", 
                                                   status="⚠️ 总结提示过长，已截取至80000字符")

                    # 使用单线程版本的请求函数
                    response = None
                    for resp in request_gpt_model_in_new_thread_with_ui_alive(
                        inputs=summary_prompt,
                        inputs_show_user="生成论文解读报告",
                        llm_kwargs=self.llm_kwargs,
                        chatbot=self.chatbot,
                        history=[],
                        sys_prompt="你是一个科研论文解读专家，请将多个方面的分析整合为一份完整、连贯、有条理的报告。报告应当重点突出，层次分明，并且保持学术性和客观性。"
                    ):
                        response = resp

                    if response:
                        self.update_detailed_progress(paper_name, "summarizing", status="✅ 总结报告生成完成")
                        return response
                    else:
                        if retry_count < self.max_retries - 1:
                            self.update_detailed_progress(paper_name, "summarizing", 
                                                       status="⚠️ 总结报告生成失败，准备重试")
                            continue
                        else:
                            self.update_detailed_progress(paper_name, "summarizing", status="❌ 总结报告生成最终失败")
                            return "报告生成失败"

                except Exception as e:
                    error_msg = str(e)
                    if retry_count < self.max_retries - 1:
                        self.update_detailed_progress(paper_name, "summarizing", 
                                                   status=f"⚠️ 总结报告生成出错，准备重试 - 错误: {error_msg}")
                        continue
                    else:
                        self.update_detailed_progress(paper_name, "summarizing", 
                                                   status=f"❌ 总结报告生成最终错误: {error_msg}")
                        return "报告生成失败: " + str(e)
            
            return "报告生成失败: 重试次数已用完"

    def save_report(self, report: str, paper_file_path: str = None) -> str:
        """保存分析报告，返回保存的文件路径"""
        paper_name = os.path.basename(paper_file_path) if paper_file_path else "未知论文"
        
        # 更新进度
        self.update_detailed_progress(paper_name, "saving", status="正在保存分析报告...")
        
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
            md_content = f"# 论文快速解读报告\n\n{report}"
            for q in self.questions:
                if q.id in self.results:
                    md_content += f"\n\n## {q.description}\n\n{self.results[q.id]}"

            result_file = write_history_to_file(
                history=[md_content],
                file_basename=f"{timestamp}_{pdf_filename}_解读报告.md"
            )

            if result_file and os.path.exists(result_file):
                promote_file_to_downloadzone(result_file, chatbot=self.chatbot)
                self.update_detailed_progress(paper_name, "saving", status=f"✅ 报告已保存: {os.path.basename(result_file)}")
                return result_file
            else:
                self.update_detailed_progress(paper_name, "saving", status="❌ 报告保存失败")
                return None
        except Exception as e:
            self.update_detailed_progress(paper_name, "saving", status=f"❌ 保存错误: {str(e)}")
            return None

    def analyze_single_paper(self, paper_path: str) -> Tuple[str, str]:
        """分析单篇论文 - 返回(论文名, 报告文件路径)"""
        try:
            # 重置分析器状态
            self.paper_content = ""
            self.results = {}
            self.paper_file_path = None
            self.current_question_index = 0
            
            paper_name = os.path.basename(paper_path)
            
            # 加载论文
            if not self._load_paper(paper_path):
                return paper_name, None

            # 分析关键问题
            for i, question in enumerate(self.questions):
                self._analyze_question(question, i)

            # 生成总结报告
            final_report = self._generate_summary()

            # 保存报告
            saved_file = self.save_report(final_report, self.paper_file_path)
            
            return paper_name, saved_file
            
        except Exception as e:
            paper_name = os.path.basename(paper_path)
            self.update_detailed_progress(paper_name, "error", status=f"❌ 分析过程出错: {str(e)}")
            return paper_name, None

    def analyze_papers_parallel(self, paper_files: List[str], max_workers: int = 3) -> List[Tuple[str, str]]:
        """并行分析多篇论文"""
        self.total_papers = len(paper_files)
        self.current_progress = 0
        
        results = []
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_paper = {
                executor.submit(self.analyze_single_paper, paper_file): paper_file 
                for paper_file in paper_files
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_paper):
                paper_file = future_to_paper[future]
                try:
                    paper_name, report_path = future.result()
                    if report_path:
                        results.append((paper_name, report_path))
                        self.update_paper_progress(paper_name, "✅ 完成")
                    else:
                        self.update_paper_progress(paper_name, "❌ 失败")
                except Exception as e:
                    paper_name = os.path.basename(paper_file)
                    self.update_paper_progress(paper_name, f"❌ 错误: {str(e)}")
        
        return results


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
    """主函数 - 批量论文速读（多线程版本）"""
    # 初始化分析器
    chatbot.append(["函数插件功能及使用方式", "批量论文速读（多线程版）：批量分析多个论文文件，为每篇论文生成独立的速读报告，支持多线程并行处理。 <br><br>📋 使用方式：<br>1、输入包含多个PDF文件的文件夹路径<br>2、或者输入多个论文ID（DOI或arXiv ID），用逗号分隔<br>3、点击插件开始批量分析"])
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

    # 创建批量分析器
    analyzer = BatchPaperAnalyzer(llm_kwargs, plugin_kwargs, chatbot, history, system_prompt)
    
    # 显示分析配置信息
    total_questions = len(analyzer.questions)
    chatbot.append(["开始批量分析", f"找到 {len(paper_files)} 篇论文，开始多线程批量分析...\n\n📋 分析配置：\n- 每篇论文分析 {total_questions} 个问题\n- 并行线程数：{min(3, len(paper_files))}\n- 预计总分析步骤：{len(paper_files) * (total_questions + 3)} 步\n\n🔄 开始处理..."])
    yield from update_ui(chatbot=chatbot, history=history)
    
    # 设置线程数（根据论文数量调整）
    max_workers = min(3, len(paper_files))  # 最多3个线程
    
    # 显示详细进度说明
    chatbot.append(["详细进度说明", f"📊 进度显示说明：\n\n🔄 处理阶段：\n- loading: 加载论文内容\n- analyzing: 分析问题 (1-{total_questions})\n- summarizing: 生成总结报告\n- saving: 保存分析报告\n\n📈 进度信息：\n- 当前论文名称\n- 当前处理阶段\n- 当前问题进度 (如适用)\n- 处理状态"])
    yield from update_ui(chatbot=chatbot, history=history)
    
    # 并行分析论文
    successful_reports = analyzer.analyze_papers_parallel(paper_files, max_workers)
    
    # 更新最终进度
    yield from update_ui(chatbot=chatbot, history=history)

    # 生成批量分析总结
    failed_count = len(paper_files) - len(successful_reports)
    summary = f"🎉 批量分析完成！\n\n"
    summary += f"📊 分析统计：\n"
    summary += f"- 总论文数：{len(paper_files)}\n"
    summary += f"- 成功分析：{len(successful_reports)}\n"
    summary += f"- 分析失败：{failed_count}\n"
    summary += f"- 并行线程：{max_workers}\n"
    summary += f"- 每篇论文分析问题：{total_questions} 个\n\n"
    
    if successful_reports:
        summary += f"✅ 成功生成报告：\n"
        for paper_name, report_path in successful_reports:
            summary += f"- {paper_name} → {os.path.basename(report_path)}\n"
    
    if failed_count > 0:
        summary += f"\n❌ 分析失败的论文：{failed_count} 篇"
    
    summary += f"\n\n💡 提示：所有报告已保存到下载区域，可直接下载使用。"

    chatbot.append(["批量分析完成", summary])
    yield from update_ui(chatbot=chatbot, history=history) 