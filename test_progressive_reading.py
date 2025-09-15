#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试递进式精读功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crazy_functions.batch_paper_detail_reading import BatchPaperDetailAnalyzer, DeepReadQuestion

def test_progressive_questions():
    """测试递进式问题设计"""
    print("=== 测试递进式精读问题设计 ===\n")
    
    # 创建分析器实例（使用模拟参数）
    mock_llm_kwargs = {'llm_model': 'gpt-3.5-turbo'}
    mock_plugin_kwargs = {}
    mock_chatbot = []
    mock_history = []
    mock_system_prompt = "测试系统提示"
    
    analyzer = BatchPaperDetailAnalyzer(
        mock_llm_kwargs, mock_plugin_kwargs, 
        mock_chatbot, mock_history, mock_system_prompt
    )
    
    print(f"总共设计了 {len(analyzer.questions)} 个递进式分析问题：\n")
    
    for i, question in enumerate(analyzer.questions, 1):
        print(f"第{i}层：{question.description}")
        print(f"重要性：{question.importance}/5")
        print(f"问题ID：{question.id}")
        print(f"问题内容：{question.question[:100]}...")
        print("-" * 80)
    
    # 验证递进关系
    print("\n=== 验证递进关系 ===")
    layer_names = [
        "问题域与动机分析",
        "理论框架与核心贡献", 
        "方法设计与技术细节",
        "实验验证与有效性分析",
        "假设条件与局限性分析",
        "复现指南与工程实现",
        "流程图与架构设计",
        "影响评估与未来展望",
        "执行摘要与要点总结"
    ]
    
    for i, (question, expected_name) in enumerate(zip(analyzer.questions, layer_names)):
        if question.description == expected_name:
            print(f"✓ 第{i+1}层：{question.description}")
        else:
            print(f"✗ 第{i+1}层：期望 '{expected_name}'，实际 '{question.description}'")
    
    print("\n=== 递进逻辑验证 ===")
    # 检查问题中是否包含递进式引用
    progressive_keywords = ["基于前面", "基于前面对", "基于前面的分析", "基于前面的技术"]
    
    for i, question in enumerate(analyzer.questions[1:], 2):  # 从第二层开始检查
        has_progressive_ref = any(keyword in question.question for keyword in progressive_keywords)
        if has_progressive_ref:
            print(f"✓ 第{i}层包含递进式引用")
        else:
            print(f"✗ 第{i}层缺少递进式引用")
    
    print("\n=== 测试完成 ===")
    return True

if __name__ == "__main__":
    test_progressive_questions()
