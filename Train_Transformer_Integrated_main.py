#----------------------------------------主执行函数和测试函数------------------------------

def test_experiment(experiment_results):
    """测试实验结果"""
    
    print(f"\n🔬 阶段4: 开始测试实验")
    start_time = time.time()
    
    config = experiment_results['config']
    save_suffix = config['save_suffix']
    save_dir = f"modelsfl{save_suffix}"
    
    try:
        # 加载训练好的模型
        transformer = TransformerPredictor(
            input_size=7,
            d_model=config['d_model'],
            nhead=config['n_heads'],
            num_layers=config['n_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            output_size=2
        ).to(device).float()
        
        # 加载模型权重
        transformer_path = f'{save_dir}/transformer_model.pth'
        state_dict = torch.load(transformer_path, map_location=device)
        # 处理DataParallel前缀
        if any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key[7:] if key.startswith('module.') else key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        transformer.load_state_dict(state_dict)
        
        # 加载MC-AE模型
        net = CombinedAE(input_size=2, encode2_input_size=3, output_size=110,
                        activation_fn=custom_activation, use_dx_in_forward=True).to(device)
        netx = CombinedAE(input_size=2, encode2_input_size=4, output_size=110, 
                         activation_fn=torch.sigmoid, use_dx_in_forward=True).to(device)
        
        net_path = f'{save_dir}/net_model.pth'
        netx_path = f'{save_dir}/netx_model.pth'
        net.load_state_dict(torch.load(net_path, map_location=device))
        netx.load_state_dict(torch.load(netx_path, map_location=device))
        
        # 加载PCA参数
        pca_params = {}
        pca_files = ['v_I', 'v', 'v_ratio', 'p_k', 'data_mean', 'data_std', 
                     'T_95_limit', 'T_99_limit', 'SPE_95_limit', 'SPE_99_limit', 
                     'P', 'k', 'P_t', 'X', 'data_nor']
        for pca_file in pca_files:
            pca_params[pca_file] = np.load(f'{save_dir}/{pca_file}.npy')
        
        print(f"✅ 模型加载完成")
        
        # 测试所有样本
        test_results = []
        
        for sample_id in ALL_TEST_SAMPLES:
            print(f"🔍 测试样本 {sample_id}...")
            
            try:
                # 加载测试数据
                vin1_data, vin2_data, vin3_data = load_test_sample(sample_id)
                
                # 处理数据格式
                if len(vin1_data.shape) == 2:
                    vin1_data = vin1_data.unsqueeze(1)
                vin1_data = vin1_data.to(torch.float32).to(device)
                
                # 定义维度
                dim_x, dim_y, dim_z, dim_q = 2, 110, 110, 3
                dim_x2, dim_y2, dim_z2, dim_q2 = 2, 110, 110, 4
                
                # 分离数据
                x_recovered = vin2_data[:, :dim_x]
                y_recovered = vin2_data[:, dim_x:dim_x + dim_y]
                z_recovered = vin2_data[:, dim_x + dim_y: dim_x + dim_y + dim_z]
                q_recovered = vin2_data[:, dim_x + dim_y + dim_z:]
                
                x_recovered2 = vin3_data[:, :dim_x2]
                y_recovered2 = vin3_data[:, dim_x2:dim_x2 + dim_y2]
                z_recovered2 = vin3_data[:, dim_x2 + dim_y2: dim_x2 + dim_y2 + dim_z2]
                q_recovered2 = vin3_data[:, dim_x2 + dim_y2 + dim_z2:]
                
                # MC-AE推理
                net.eval()
                netx.eval()
                
                with torch.no_grad():
                    net = net.double()
                    netx = netx.double()
                    
                    recon_imtest = net(x_recovered, z_recovered, q_recovered)
                    reconx_imtest = netx(x_recovered2, z_recovered2, q_recovered2)
                
                # 计算重构误差
                AA = recon_imtest[0].cpu().detach().numpy()
                yTrainU = y_recovered.cpu().detach().numpy()
                ERRORU = AA - yTrainU
            
                BB = reconx_imtest[0].cpu().detach().numpy()
                yTrainX = y_recovered2.cpu().detach().numpy()
                ERRORX = BB - yTrainX
            
                # 诊断特征提取
                df_data = DiagnosisFeature(ERRORU, ERRORX)
                
                # 使用预训练的PCA参数进行综合计算
                time_axis = np.arange(df_data.shape[0])
                
                lamda, CONTN, t_total, q_total, S, FAI, g, h, kesi, fai, f_time, level, maxlevel, contTT, contQ, X_ratio, CContn, data_mean, data_std = Comprehensive_calculation(
                    df_data.values, 
                    pca_params['data_mean'], 
                    pca_params['data_std'], 
                    pca_params['v'].reshape(len(pca_params['v']), 1), 
                    pca_params['p_k'], 
                    pca_params['v_I'], 
                    pca_params['T_99_limit'], 
                    pca_params['SPE_99_limit'], 
                    pca_params['X'], 
                    time_axis
                )
                
                # 计算阈值
                threshold1, threshold2, threshold3 = calculate_thresholds(fai)
                
                # 三窗口故障检测
                fault_labels, detection_info = three_window_fault_detection(fai, threshold1, sample_id)
                
                # 构建结果
                sample_result = {
                    'sample_id': sample_id,
                    'config_name': config['name'],
                    'label': 1 if sample_id in TEST_SAMPLES['fault'] else 0,
                    'df_data': df_data.values,
                    'fai': fai,
                    'T_squared': t_total,
                    'SPE': q_total,
                    'thresholds': {
                        'threshold1': threshold1,
                        'threshold2': threshold2, 
                        'threshold3': threshold3
                    },
                    'fault_labels': fault_labels,
                    'detection_info': detection_info,
                    'performance_metrics': {
                        'fai_mean': np.mean(fai),
                        'fai_std': np.std(fai),
                        'fai_max': np.max(fai),
                        'fai_min': np.min(fai),
                        'anomaly_count': np.sum(fai > threshold1),
                        'anomaly_ratio': np.sum(fai > threshold1) / len(fai)
                    }
                }
                
                test_results.append(sample_result)
                
                # 输出简要结果
                metrics = sample_result['performance_metrics']
                window_stats = detection_info.get('window_stats', {})
                print(f"   样本{sample_id}: fai均值={metrics['fai_mean']:.6f}, "
                      f"异常率={metrics['anomaly_ratio']:.2%}, "
                      f"故障率={window_stats.get('fault_ratio', 0.0):.2%}")
                
            except Exception as e:
                print(f"❌ 样本 {sample_id} 测试失败: {e}")
                continue
        
        # 计算性能指标
        performance_metrics = calculate_performance_metrics(test_results)
        
        # 记录测试结果
        experiment_results['test_results'] = {
            'sample_results': test_results,
            'performance_metrics': performance_metrics
        }
        experiment_results['timing']['test'] = time.time() - start_time
        experiment_results['stage'] = 'test_completed'
        
        print(f"✅ 测试完成，用时: {experiment_results['timing']['test']:.2f}秒")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise e
    
    return experiment_results

def calculate_performance_metrics(test_results):
    """计算性能指标"""
    
    # 收集所有样本的预测结果
    all_true_labels = []
    all_fai_values = []
    all_fault_predictions = []
    
    for result in test_results:
        true_label = result.get('label', 0)
        fai_values = result.get('fai', [])
        fault_labels = result.get('fault_labels', [])
        thresholds = result.get('thresholds', {})
        threshold1 = thresholds.get('threshold1', 0.0)
        
        # 对于每个时间点
        for i, (fai_val, fault_pred) in enumerate(zip(fai_values, fault_labels)):
            all_true_labels.append(true_label)
            all_fai_values.append(fai_val)
            
            # 根据ROC逻辑计算预测结果
            if true_label == 0:  # 正常样本
                prediction = 1 if fai_val > threshold1 else 0
            else:  # 故障样本
                if fai_val > threshold1 and fault_pred == 1:
                    prediction = 1  # TP
                else:
                    prediction = 0  # FN
            
            all_fault_predictions.append(prediction)
    
    # 计算ROC指标
    all_true_labels = np.array(all_true_labels)
    all_fai_values = np.array(all_fai_values)
    all_fault_predictions = np.array(all_fault_predictions)
    
    # 计算混淆矩阵
    tn = np.sum((all_true_labels == 0) & (all_fault_predictions == 0))
    fp = np.sum((all_true_labels == 0) & (all_fault_predictions == 1))
    fn = np.sum((all_true_labels == 1) & (all_fault_predictions == 0))
    tp = np.sum((all_true_labels == 1) & (all_fault_predictions == 1))
    
    # 计算性能指标
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    tpr = recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # 样本级统计
    sample_metrics = {
        'total_samples': len(test_results),
        'normal_samples': len([r for r in test_results if r['label'] == 0]),
        'fault_samples': len([r for r in test_results if r['label'] == 1]),
        'avg_fai_normal': np.mean([r['performance_metrics']['fai_mean'] 
                                 for r in test_results if r['label'] == 0]),
        'avg_fai_fault': np.mean([r['performance_metrics']['fai_mean'] 
                                for r in test_results if r['label'] == 1]),
        'avg_anomaly_ratio_normal': np.mean([r['performance_metrics']['anomaly_ratio'] 
                                           for r in test_results if r['label'] == 0]),
        'avg_anomaly_ratio_fault': np.mean([r['performance_metrics']['anomaly_ratio'] 
                                          for r in test_results if r['label'] == 1])
    }
    
    return {
        'confusion_matrix': {'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)},
        'classification_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'specificity': float(specificity),
            'tpr': float(tpr),
            'fpr': float(fpr)
        },
        'sample_metrics': sample_metrics,
        'roc_data': {
            'true_labels': all_true_labels.tolist(),
            'fai_values': all_fai_values.tolist(),
            'predictions': all_fault_predictions.tolist()
        }
    }

def create_comparison_visualization(results_original, results_enhanced, save_dir):
    """创建对比可视化图表"""
    
    print(f"🎨 生成对比可视化图表...")
    
    # 创建可视化目录
    viz_dir = f"{save_dir}/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # 图1: ROC曲线对比
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 子图1: ROC曲线对比
    perf_orig = results_original['test_results']['performance_metrics']
    perf_enh = results_enhanced['test_results']['performance_metrics']
    
    # 绘制工作点
    ax1.scatter(perf_orig['classification_metrics']['fpr'], 
               perf_orig['classification_metrics']['tpr'], 
               s=200, color='blue', marker='o', 
               label=f"原始参数 (AUC估算)", alpha=0.8)
    ax1.scatter(perf_enh['classification_metrics']['fpr'], 
               perf_enh['classification_metrics']['tpr'], 
               s=200, color='red', marker='^', 
               label=f"增强参数 (AUC估算)", alpha=0.8)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机分类器')
    
    if use_chinese:
        ax1.set_xlabel('假正例率 (FPR)')
        ax1.set_ylabel('真正例率 (TPR)')
        ax1.set_title('(a) ROC曲线对比')
    else:
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('(a) ROC Curve Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 性能指标对比
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    orig_values = [perf_orig['classification_metrics']['accuracy'],
                   perf_orig['classification_metrics']['precision'],
                   perf_orig['classification_metrics']['recall'],
                   perf_orig['classification_metrics']['f1_score'],
                   perf_orig['classification_metrics']['specificity']]
    enh_values = [perf_enh['classification_metrics']['accuracy'],
                  perf_enh['classification_metrics']['precision'],
                  perf_enh['classification_metrics']['recall'],
                  perf_enh['classification_metrics']['f1_score'],
                  perf_enh['classification_metrics']['specificity']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, orig_values, width, label='原始参数', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, enh_values, width, label='增强参数', color='red', alpha=0.7)
    
    if use_chinese:
        ax2.set_xlabel('性能指标')
        ax2.set_ylabel('分数')
        ax2.set_title('(b) 性能指标对比')
    else:
        ax2.set_xlabel('Performance Metrics')
        ax2.set_ylabel('Score')
        ax2.set_title('(b) Performance Metrics Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 子图3: 训练损失对比
    train_losses_orig = results_original['transformer_results']['train_losses']
    train_losses_enh = results_enhanced['transformer_results']['train_losses']
    
    ax3.plot(train_losses_orig, 'b-', label='原始参数', linewidth=2)
    ax3.plot(train_losses_enh, 'r-', label='增强参数', linewidth=2)
    
    if use_chinese:
        ax3.set_xlabel('训练轮数')
        ax3.set_ylabel('训练损失')
        ax3.set_title('(c) 训练损失对比')
    else:
        ax3.set_xlabel('Training Epochs')
        ax3.set_ylabel('Training Loss')
        ax3.set_title('(c) Training Loss Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 子图4: 样本级性能对比
    sample_metrics_orig = perf_orig['sample_metrics']
    sample_metrics_enh = perf_enh['sample_metrics']
    
    sample_names = ['正常样本\nFAI均值', '故障样本\nFAI均值', '正常样本\n异常率', '故障样本\n异常率']
    orig_sample_values = [sample_metrics_orig['avg_fai_normal'],
                         sample_metrics_orig['avg_fai_fault'],
                         sample_metrics_orig['avg_anomaly_ratio_normal'],
                         sample_metrics_orig['avg_anomaly_ratio_fault']]
    enh_sample_values = [sample_metrics_enh['avg_fai_normal'],
                        sample_metrics_enh['avg_fai_fault'],
                        sample_metrics_enh['avg_anomaly_ratio_normal'],
                        sample_metrics_enh['avg_anomaly_ratio_fault']]
    
    x = np.arange(len(sample_names))
    bars1 = ax4.bar(x - width/2, orig_sample_values, width, label='原始参数', color='blue', alpha=0.7)
    bars2 = ax4.bar(x + width/2, enh_sample_values, width, label='增强参数', color='red', alpha=0.7)
    
    if use_chinese:
        ax4.set_xlabel('样本级指标')
        ax4.set_ylabel('数值')
        ax4.set_title('(d) 样本级性能对比')
    else:
        ax4.set_xlabel('Sample-level Metrics')
        ax4.set_ylabel('Value')
        ax4.set_title('(d) Sample-level Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(sample_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = f"{viz_dir}/experiment_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 对比图表已保存: {comparison_path}")
    
    return comparison_path

def main():
    """主执行函数"""
    
    print("\n🚀 开始集成训练测试实验...")
    
    all_results = {}
    
    # 执行两个实验配置
    for config_name, config in EXPERIMENT_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"🔬 实验配置: {config['name']}")
        print(f"{'='*80}")
        
        try:
            # 训练实验
            experiment_results = train_experiment(config_name, config)
            
            # 立即测试
            experiment_results = test_experiment(experiment_results)
            
            # 标记完成
            experiment_results['stage'] = 'completed'
            
            # 保存最终结果
            save_suffix = config['save_suffix']
            save_dir = f"modelsfl{save_suffix}"
            checkpoint_path = f"{save_dir}/checkpoint.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(experiment_results, f)
            
            # 保存到全局结果
            all_results[config_name] = experiment_results
            
            # 输出实验总结
            print(f"\n📊 {config['name']} 实验总结:")
            print(f"   训练时间: {experiment_results['timing']['transformer']:.2f}秒")
            print(f"   测试时间: {experiment_results['timing']['test']:.2f}秒")
            
            perf = experiment_results['test_results']['performance_metrics']
            metrics = perf['classification_metrics']
            print(f"   准确率: {metrics['accuracy']:.3f}")
            print(f"   精确率: {metrics['precision']:.3f}")
            print(f"   召回率: {metrics['recall']:.3f}")
            print(f"   F1分数: {metrics['f1_score']:.3f}")
            print(f"   TPR: {metrics['tpr']:.3f}, FPR: {metrics['fpr']:.3f}")
            
        except Exception as e:
            print(f"❌ 实验 {config['name']} 失败: {e}")
            continue
    
    # 如果两个实验都完成，生成对比可视化
    if len(all_results) == 2:
        print(f"\n🎨 生成对比分析...")
        try:
            comparison_path = create_comparison_visualization(
                all_results['original'], 
                all_results['enhanced'],
                "modelsfl_comparison"
            )
            
            # 保存完整结果
            results_path = "modelsfl_comparison/complete_results.pkl"
            os.makedirs("modelsfl_comparison", exist_ok=True)
            with open(results_path, 'wb') as f:
                pickle.dump(all_results, f)
            
            print(f"✅ 完整结果已保存: {results_path}")
            
        except Exception as e:
            print(f"❌ 对比分析失败: {e}")
    
    # 最终总结
    print(f"\n{'='*80}")
    print(f"🎉 集成训练测试实验完成！")
    print(f"{'='*80}")
    
    for config_name, results in all_results.items():
        config = EXPERIMENT_CONFIGS[config_name]
        print(f"\n📊 {config['name']} 最终结果:")
        
        if 'test_results' in results:
            perf = results['test_results']['performance_metrics']
            metrics = perf['classification_metrics']
            print(f"   ROC工作点: TPR={metrics['tpr']:.3f}, FPR={metrics['fpr']:.3f}")
            print(f"   综合性能: F1={metrics['f1_score']:.3f}")
            print(f"   结果保存: modelsfl{config['save_suffix']}/")
        else:
            print(f"   ❌ 实验未完成")
    
    print(f"\n🔄 下一步可以:")
    print(f"   1. 查看详细的训练和测试结果")
    print(f"   2. 分析两种参数配置的性能差异")
    print(f"   3. 检查反向传播机制的效果")
    print(f"   4. 优化模型参数和训练策略")

if __name__ == "__main__":
    main()