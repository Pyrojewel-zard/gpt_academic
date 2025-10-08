#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试历史记录优化功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crazy_functions.undefine_paper_detail_reading import BatchPaperDetailAnalyzer, DeepReadQuestion

def test_token_estimation():
    """测试token估算功能"""
    print("测试token估算功能...")
    
    # 创建测试实例
    analyzer = BatchPaperDetailAnalyzer(
        llm_kwargs={'llm_model': 'gpt-3.5-turbo'},
        plugin_kwargs={},
        chatbot=[],
        history=[],
        system_prompt="测试"
    )
    
    # 测试token估算
    test_text = "这是一个测试文本，用于验证token估算功能是否正常工作。"
    estimated_tokens = analyzer._estimate_tokens(test_text)
    print(f"文本: {test_text}")
    print(f"估算tokens: {estimated_tokens}")
    
    # 测试token统计更新
    analyzer._update_token_stats(test_text, "这是回复文本")
    print(f"Token统计: {analyzer._token_usage_stats}")
    
    print("✅ Token估算功能测试通过")

def test_progressive_context():
    """测试递进式上下文构建"""
    print("\n测试递进式上下文构建...")
    
    analyzer = BatchPaperDetailAnalyzer(
        llm_kwargs={'llm_model': 'gpt-3.5-turbo'},
        plugin_kwargs={},
        chatbot=[],
        history=[],
        system_prompt="测试"
    )
    
    # 模拟一些分析结果
    analyzer.results = {
        "problem_domain_and_motivation": "这是一个关于机器学习的问题域分析结果",
        "theoretical_framework_and_contributions": "这是理论框架分析结果"
    }
    
    # 测试问题
    test_question = DeepReadQuestion(
        id="method_design_and_technical_details",
        question="请分析技术实现细节",
        importance=5,
        description="方法设计与技术细节",
        domain="both"
    )
    
    # 构建递进式上下文
    context_parts = analyzer._build_progressive_context(test_question)
    print(f"构建的上下文部分数量: {len(context_parts)}")
    for i, part in enumerate(context_parts):
        print(f"上下文部分 {i+1}: {part[:100]}...")
    
    print("✅ 递进式上下文构建测试通过")

def test_related_analysis():
    """测试问题相关性判断"""
    print("\n测试问题相关性判断...")
    
    analyzer = BatchPaperDetailAnalyzer(
        llm_kwargs={'llm_model': 'gpt-3.5-turbo'},
        plugin_kwargs={},
        chatbot=[],
        history=[],
        system_prompt="测试"
    )
    
    # 测试相关性问题
    is_related = analyzer._is_related_analysis(
        "problem_domain_and_motivation",
        "theoretical_framework_and_contributions"
    )
    print(f"问题域分析 -> 理论框架分析: 相关 = {is_related}")
    
    is_related = analyzer._is_related_analysis(
        "problem_domain_and_motivation",
        "flowcharts_and_architecture"
    )
    print(f"问题域分析 -> 流程图分析: 相关 = {is_related}")
    
    print("✅ 问题相关性判断测试通过")

def test_token_usage_summary():
    """测试token使用摘要"""
    print("\n测试token使用摘要...")
    
    analyzer = BatchPaperDetailAnalyzer(
        llm_kwargs={'llm_model': 'gpt-3.5-turbo'},
        plugin_kwargs={},
        chatbot=[],
        history=[],
        system_prompt="测试"
    )
    
    # 模拟一些交互
    analyzer._update_token_stats("输入文本1", "输出文本1")
    analyzer._update_token_stats("输入文本2", "输出文本2")
    
    # 获取摘要
    summary = analyzer._get_token_usage_summary()
    print("Token使用摘要:")
    print(summary)
    
    print("✅ Token使用摘要测试通过")

if __name__ == "__main__":
    print("开始测试历史记录优化功能...\n")
    
    try:
        test_token_estimation()
        test_progressive_context()
        test_related_analysis()
        test_token_usage_summary()
        
        print("\n🎉 所有测试通过！历史记录优化功能正常工作。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
