#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新添加的Transformer模型对比功能
验证雷达图和ROC分析图的生成
"""

import os
import sys
import tempfile
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Linux环境配置
mpl.use('Agg')

def test_transformer_comparison():
    """测试Transformer对比功能"""
    print("🧪 Testing Transformer comparison functionality...")
    
    # 导入修改后的可视化模块
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from Run_Complete_Visualization import CompleteVisualizationRunner
        
        # 使用临时目录进行测试
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Using temporary directory: {temp_dir}")
            
            # 创建可视化运行器实例
            runner = CompleteVisualizationRunner(base_dir=temp_dir)
            
            # 测试单独的Transformer对比功能
            print("\n🔄 Testing _run_transformer_comparison_analysis()...")
            result = runner._run_transformer_comparison_analysis()
            
            if result['status'] == 'success':
                print("✅ Transformer comparison analysis succeeded")
                
                # 检查输出文件
                output_file = result.get('transformer_comparison_charts')
                if output_file and os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    print(f"✅ Output file generated: {os.path.basename(output_file)} ({file_size} bytes)")
                    
                    # 验证图像文件完整性
                    if file_size > 0:
                        print("✅ Output file has valid size")
                    else:
                        print("❌ Output file is empty")
                        return False
                else:
                    print("❌ Output file not found")
                    return False
            else:
                print(f"❌ Transformer comparison analysis failed: {result.get('error', 'Unknown error')}")
                return False
            
            # 测试完整分析流程中的集成
            print("\n🔄 Testing integration in complete analysis...")
            try:
                analysis_results, report_path = runner.run_complete_analysis()
                
                if 'transformer_comparison' in analysis_results:
                    comp_result = analysis_results['transformer_comparison']
                    if comp_result and comp_result.get('status') == 'success':
                        print("✅ Transformer comparison integrated successfully in complete analysis")
                    else:
                        print("⚠️  Transformer comparison had issues in complete analysis")
                else:
                    print("❌ Transformer comparison not found in complete analysis results")
                    return False
                    
            except Exception as e:
                print(f"❌ Complete analysis integration test failed: {e}")
                return False
            
            print("\n🎉 All tests passed!")
            return True
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_individual_chart_functions():
    """测试单独的图表生成函数"""
    print("\n🧪 Testing individual chart generation functions...")
    
    try:
        from Run_Complete_Visualization import CompleteVisualizationRunner
        
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = CompleteVisualizationRunner(base_dir=temp_dir)
            
            # 模拟性能指标数据
            metrics1 = {
                'accuracy': 0.92, 'precision': 0.88, 'recall': 0.95,
                'f1_score': 0.91, 'specificity': 0.89, 'tpr': 0.95, 'fpr': 0.11, 'auc': 0.94
            }
            metrics2 = {
                'accuracy': 0.94, 'precision': 0.91, 'recall': 0.93,
                'f1_score': 0.92, 'specificity': 0.95, 'tpr': 0.93, 'fpr': 0.05, 'auc': 0.96
            }
            
            # 测试各个图表生成函数
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 测试雷达图
            try:
                ax1 = plt.subplot(2, 3, 1, projection='polar')
                runner._create_radar_comparison(ax1, metrics1, metrics2)
                print("✅ Radar chart generation works")
            except Exception as e:
                print(f"❌ Radar chart failed: {e}")
                return False
            
            # 测试ROC曲线
            try:
                ax2 = plt.subplot(2, 3, 2)
                runner._create_roc_comparison(ax2, metrics1, metrics2)
                print("✅ ROC curve generation works")
            except Exception as e:
                print(f"❌ ROC curve failed: {e}")
                return False
            
            # 测试性能指标条形图
            try:
                ax3 = plt.subplot(2, 3, 3)
                runner._create_metrics_comparison_bar(ax3, metrics1, metrics2)
                print("✅ Metrics bar chart generation works")
            except Exception as e:
                print(f"❌ Metrics bar chart failed: {e}")
                return False
            
            # 测试工作点对比
            try:
                ax4 = plt.subplot(2, 3, 4)
                runner._create_working_point_comparison(ax4, metrics1, metrics2)
                print("✅ Working point comparison works")
            except Exception as e:
                print(f"❌ Working point comparison failed: {e}")
                return False
            
            # 测试精度-召回率曲线
            try:
                ax5 = plt.subplot(2, 3, 5)
                runner._create_precision_recall_comparison(ax5, metrics1, metrics2)
                print("✅ Precision-recall curve generation works")
            except Exception as e:
                print(f"❌ Precision-recall curve failed: {e}")
                return False
            
            # 测试混淆矩阵
            try:
                ax6 = plt.subplot(2, 3, 6)
                runner._create_confusion_matrix_comparison(ax6)
                print("✅ Confusion matrix comparison works")
            except Exception as e:
                print(f"❌ Confusion matrix comparison failed: {e}")
                return False
            
            # 保存测试图表
            test_output = os.path.join(temp_dir, "test_charts.png")
            plt.tight_layout()
            plt.savefig(test_output, dpi=150, bbox_inches='tight')
            plt.close()
            
            if os.path.exists(test_output) and os.path.getsize(test_output) > 0:
                print(f"✅ Test chart saved successfully: {os.path.basename(test_output)}")
            else:
                print("❌ Test chart save failed")
                return False
            
            print("🎉 All individual chart functions work correctly!")
            return True
            
    except Exception as e:
        print(f"❌ Individual chart test failed: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 Starting Transformer Comparison Feature Tests")
    print("="*60)
    
    # 测试1: 整体功能测试
    test1_passed = test_transformer_comparison()
    
    # 测试2: 单独函数测试
    test2_passed = test_individual_chart_functions()
    
    # 总结测试结果
    print("\n📊 Test Summary:")
    print("="*60)
    print(f"🧪 Overall functionality test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"🔧 Individual functions test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests PASSED! The new Transformer comparison feature is working correctly.")
        return True
    else:
        print("\n❌ Some tests FAILED. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
