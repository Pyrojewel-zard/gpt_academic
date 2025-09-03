import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crazy_functions.Batch_Paper_Reading import BatchPaperAnalyzer, _find_paper_files, PaperQuestion


class TestBatchPaperReading(unittest.TestCase):
    """测试批量论文速读功能"""

    def setUp(self):
        """设置测试环境"""
        self.llm_kwargs = {"llm_model": "gpt-3.5-turbo"}
        self.plugin_kwargs = {}
        self.chatbot = []
        self.history = []
        self.system_prompt = "测试系统提示"

    def test_paper_question_structure(self):
        """测试论文问题结构"""
        question = PaperQuestion(
            id="test_id",
            question="测试问题",
            importance=5,
            description="测试描述"
        )
        
        self.assertEqual(question.id, "test_id")
        self.assertEqual(question.question, "测试问题")
        self.assertEqual(question.importance, 5)
        self.assertEqual(question.description, "测试描述")

    def test_batch_analyzer_initialization(self):
        """测试批量分析器初始化"""
        analyzer = BatchPaperAnalyzer(
            self.llm_kwargs, 
            self.plugin_kwargs, 
            self.chatbot, 
            self.history, 
            self.system_prompt
        )
        
        self.assertIsNotNone(analyzer.questions)
        self.assertEqual(len(analyzer.questions), 4)
        # 检查问题是否按重要性排序
        self.assertEqual(analyzer.questions[0].importance, 5)
        self.assertEqual(analyzer.questions[-1].importance, 2)

    def test_find_paper_files_single_file(self):
        """测试查找单个论文文件"""
        # 创建一个临时测试文件
        test_file = "test_paper.pdf"
        with open(test_file, 'w') as f:
            f.write("test content")
        
        try:
            files = _find_paper_files(test_file)
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0], test_file)
        finally:
            # 清理测试文件
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_find_paper_files_unsupported_format(self):
        """测试不支持的格式"""
        test_file = "test_file.xyz"
        with open(test_file, 'w') as f:
            f.write("test content")
        
        try:
            files = _find_paper_files(test_file)
            self.assertEqual(len(files), 0)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_find_paper_files_directory(self):
        """测试在目录中查找论文文件"""
        # 创建临时测试目录
        test_dir = "test_papers"
        os.makedirs(test_dir, exist_ok=True)
        
        # 创建测试文件
        test_files = [
            "paper1.pdf",
            "paper2.docx", 
            "paper3.txt",
            "not_a_paper.xyz"
        ]
        
        for file_name in test_files:
            with open(os.path.join(test_dir, file_name), 'w') as f:
                f.write("test content")
        
        try:
            files = _find_paper_files(test_dir)
            # 应该找到3个支持的格式文件
            self.assertEqual(len(files), 3)
            # 检查是否包含所有支持的格式
            extensions = [os.path.splitext(f)[1].lower() for f in files]
            self.assertIn('.pdf', extensions)
            self.assertIn('.docx', extensions)
            self.assertIn('.txt', extensions)
        finally:
            # 清理测试目录
            import shutil
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

    def test_save_report_filename_generation(self):
        """测试报告文件名生成"""
        analyzer = BatchPaperAnalyzer(
            self.llm_kwargs, 
            self.plugin_kwargs, 
            self.chatbot, 
            self.history, 
            self.system_prompt
        )
        
        # 测试文件名清理
        test_path = "test/path/paper with spaces & special chars.pdf"
        analyzer.paper_file_path = test_path
        
        # 模拟结果
        analyzer.results = {
            "research_and_methods": "测试结果1",
            "findings_and_innovation": "测试结果2"
        }
        
        # 测试保存报告（不实际保存文件）
        with patch('crazy_functions.Batch_Paper_Reading.write_history_to_file') as mock_write:
            mock_write.return_value = "test_report.md"
            with patch('crazy_functions.Batch_Paper_Reading.promote_file_to_downloadzone'):
                with patch('os.path.exists', return_value=True):
                    result = analyzer.save_report("测试报告内容")
                    self.assertEqual(result, "test_report.md")


if __name__ == '__main__':
    unittest.main()
