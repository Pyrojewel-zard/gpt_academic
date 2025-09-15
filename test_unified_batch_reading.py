#!/usr/bin/env python3
"""
统一批量论文阅读器测试脚本
测试主题分类和动态prompt功能
"""

import os
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_unified_analyzer():
    """测试统一分析器的基本功能"""
    try:
        from crazy_functions.unified_batch_paper_reading import UnifiedBatchPaperAnalyzer, PaperQuestion
        
        print("✅ 成功导入 UnifiedBatchPaperAnalyzer")
        
        # 模拟参数
        llm_kwargs = {'llm_model': 'gpt-3.5-turbo'}
        plugin_kwargs = {}
        chatbot = []
        history = []
        system_prompt = "测试系统提示"
        
        # 创建分析器实例
        analyzer = UnifiedBatchPaperAnalyzer(llm_kwargs, plugin_kwargs, chatbot, history, system_prompt)
        
        print("✅ 成功创建分析器实例")
        
        # 测试问题分类
        general_questions = [q for q in analyzer.questions if q.domain in ["both", "general"]]
        rf_ic_questions = [q for q in analyzer.questions if q.domain in ["both", "rf_ic"]]
        
        print(f"✅ 通用问题数量: {len(general_questions)}")
        print(f"✅ RF IC问题数量: {len(rf_ic_questions)}")
        
        # 测试领域特定问题获取
        analyzer.paper_domain = "general"
        general_domain_questions = analyzer._get_domain_specific_questions()
        print(f"✅ 通用领域问题数量: {len(general_domain_questions)}")
        
        analyzer.paper_domain = "rf_ic"
        rf_ic_domain_questions = analyzer._get_domain_specific_questions()
        print(f"✅ RF IC领域问题数量: {len(rf_ic_domain_questions)}")
        
        # 测试系统提示生成
        general_sys_prompt = analyzer._get_domain_specific_system_prompt()
        analyzer.paper_domain = "rf_ic"
        rf_ic_sys_prompt = analyzer._get_domain_specific_system_prompt()
        
        print("✅ 通用系统提示:", general_sys_prompt[:50] + "...")
        print("✅ RF IC系统提示:", rf_ic_sys_prompt[:50] + "...")
        
        # 测试分析提示生成
        test_question = analyzer.questions[0]  # 取第一个问题
        analyzer.paper_domain = "general"
        general_prompt = analyzer._get_domain_specific_analysis_prompt(test_question)
        analyzer.paper_domain = "rf_ic"
        rf_ic_prompt = analyzer._get_domain_specific_analysis_prompt(test_question)
        
        print("✅ 通用分析提示长度:", len(general_prompt))
        print("✅ RF IC分析提示长度:", len(rf_ic_prompt))
        
        print("\n🎉 所有测试通过！统一分析器功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_question_structure():
    """测试问题结构"""
    try:
        from crazy_functions.unified_batch_paper_reading import PaperQuestion
        
        # 创建测试问题
        question = PaperQuestion(
            id="test_question",
            question="这是一个测试问题？",
            importance=5,
            description="测试问题描述",
            domain="both"
        )
        
        print("✅ 问题结构测试通过")
        print(f"   - ID: {question.id}")
        print(f"   - 重要性: {question.importance}")
        print(f"   - 领域: {question.domain}")
        
        return True
        
    except Exception as e:
        print(f"❌ 问题结构测试失败: {str(e)}")
        return False

def test_import_compatibility():
    """测试导入兼容性"""
    try:
        # 测试是否能导入原有模块
        from crazy_functions.Batch_Paper_Reading import BatchPaperAnalyzer
        from crazy_functions.batch_rf_ic_reading import BatchRFICAnalyzer
        
        print("✅ 原有模块导入正常，兼容性良好")
        
        # 测试新模块
        from crazy_functions.unified_batch_paper_reading import UnifiedBatchPaperAnalyzer
        
        print("✅ 新统一模块导入正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入兼容性测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 开始测试统一批量论文阅读器...")
    print("=" * 50)
    
    tests = [
        ("导入兼容性测试", test_import_compatibility),
        ("问题结构测试", test_question_structure),
        ("统一分析器功能测试", test_unified_analyzer),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} 失败")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！统一批量论文阅读器可以正常使用")
    else:
        print("⚠️  部分测试失败，请检查代码")

