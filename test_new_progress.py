#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新的进度条系统
"""

def test_paper_progress_bar():
    """测试单篇论文进度条"""
    print("=== 测试单篇论文进度条 ===")
    
    # 模拟4个问题
    total_questions = 4
    question_descriptions = [
        "研究问题与方法",
        "研究发现与创新", 
        "研究方法与数据",
        "局限性与影响"
    ]
    
    # 创建进度条
    from crazy_functions.Batch_Paper_Reading import PaperProgressBar
    
    progress_bar = PaperProgressBar("测试论文.pdf", total_questions)
    
    print(f"初始状态: {progress_bar.get_current_progress()}")
    
    # 模拟处理每个问题
    for i, desc in enumerate(question_descriptions):
        print(f"\n处理问题 {i+1}: {desc}")
        progress_str = progress_bar.update_question(desc, "分析中")
        print(f"进度: {progress_str}")
        print(f"当前状态: {progress_bar.current_status}")
    
    print(f"\n最终进度: {progress_bar.get_current_progress()}")


def test_batch_progress_tracker():
    """测试批量进度跟踪器"""
    print("\n\n=== 测试批量进度跟踪器 ===")
    
    from crazy_functions.Batch_Paper_Reading import BatchProgressTracker
    
    # 模拟3篇论文，每篇4个问题
    tracker = BatchProgressTracker(3, 4)
    
    # 添加论文
    papers = ["论文A.pdf", "论文B.pdf", "论文C.pdf"]
    for paper in papers:
        tracker.add_paper(paper)
    
    print("初始状态:")
    for paper_name, progress_bar in tracker.paper_progress_bars.items():
        print(f"  {paper_name}: {progress_bar.get_current_progress()} - {progress_bar.current_status}")
    
    # 模拟处理论文A的第一个问题
    print("\n处理论文A的第一个问题:")
    tracker.update_paper_question("论文A.pdf", "研究问题与方法", "分析中")
    
    for paper_name, progress_bar in tracker.paper_progress_bars.items():
        print(f"  {paper_name}: {progress_bar.get_current_progress()} - {progress_bar.current_status}")
    
    # 模拟完成论文A
    print("\n完成论文A:")
    tracker.complete_paper("论文A.pdf")
    
    for paper_name, progress_bar in tracker.paper_progress_bars.items():
        status_icon = "✅" if progress_bar.current_question >= progress_bar.total_questions else "🔄"
        print(f"  {status_icon} {paper_name}: {progress_bar.get_current_progress()} - {progress_bar.current_status}")
    
    print(f"\n已完成论文数: {tracker.completed_papers}/{tracker.total_papers}")


def test_progress_display():
    """测试进度显示格式"""
    print("\n\n=== 测试进度显示格式 ===")
    
    # 模拟进度显示消息
    progress_msg = f"📊 批量论文分析进度\n\n"
    progress_msg += f"**整体进度**: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.0% (预计剩余: 计算中...)\n"
    progress_msg += f"**已完成论文**: 0/3\n\n"
    progress_msg += f"**各论文进度**:\n"
    
    papers = ["论文A.pdf", "论文B.pdf", "论文C.pdf"]
    for paper in papers:
        progress_msg += f"🔄 **{paper}**: [░░░░░░░░░░░░░░░░░░░░░] 0.0% - 准备中\n"
    
    progress_msg += f"\n📋 **分析统计**:\n"
    progress_msg += f"- 成功: 0 篇\n"
    progress_msg += f"- 失败: 0 篇\n"
    progress_msg += f"- 错误: 0 篇\n\n"
    progress_msg += f"**支持格式**: PDF, DOCX, DOC, TXT, MD, TEX\n"
    progress_msg += f"**并行线程**: 3 个"
    
    print(progress_msg)


if __name__ == "__main__":
    try:
        test_paper_progress_bar()
        test_batch_progress_tracker()
        test_progress_display()
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保在正确的环境中运行此测试")
