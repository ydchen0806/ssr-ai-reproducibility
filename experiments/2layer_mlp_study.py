#!/usr/bin/env python3
"""
2层MLP连续学习实验 - 7组消融实验
支持两种评估协议:
- CIL (Class-Incremental): 单头分类器，测试所有100类
- TIL (Task-Incremental): 单头分类器+掩码屏蔽，只测试当前task的5类

实验配置:
1. Baseline: 无正则化
2. MAS (Full): MAS保护所有参数
3. EWC (Full): EWC保护所有参数
4. EWC + SSR: Hidden用EWC, Classifier用SSR
5. MAS + SSR: Hidden用MAS, Classifier用SSR
6. SI + SSR: Hidden用SI, Classifier用SSR
7. SI (Full): SI保护所有参数
"""

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import os
import time
from typing import Dict, Tuple, List
from datetime import datetime

from models.two_layer_mlp import TwoLayerMLP, FourLayerMLP, create_mlp_model
from regularizers.layer_wise import LayerWiseRegularizer
from regularizers.si import SI
from cifar100_feature_dataset import SuperclassSplitCIFAR100

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_task_mask(task_id: int, fine_to_superclass: np.ndarray, 
                  superclass_order: List[int], num_classes: int = 100, 
                  device: str = 'cuda') -> torch.Tensor:
    """
    获取任务掩码，用于TIL评估
    
    Args:
        task_id: 当前任务ID
        fine_to_superclass: fine label到superclass的映射
        superclass_order: superclass的顺序
        num_classes: 总类别数
        device: 设备
    
    Returns:
        mask: 布尔张量，属于当前任务的类别为True
    """
    target_superclass = superclass_order[task_id]
    # 创建掩码：属于当前task的fine labels为True
    mask = torch.tensor([fine_to_superclass[i] == target_superclass 
                         for i in range(num_classes)], 
                        dtype=torch.bool, device=device)
    return mask


def apply_task_mask(logits: torch.Tensor, task_id: int, 
                    fine_to_superclass: np.ndarray, 
                    superclass_order: List[int]) -> torch.Tensor:
    """
    应用任务掩码，将非当前任务的logits设为-1e9
    
    Args:
        logits: 模型输出 (batch_size, 100)
        task_id: 当前任务ID
        fine_to_superclass: fine label到superclass的映射
        superclass_order: superclass的顺序
    
    Returns:
        masked_logits: 掩码后的logits
    """
    mask = get_task_mask(task_id, fine_to_superclass, superclass_order, 
                        num_classes=logits.size(1), device=logits.device)
    # 创建掩码后的logits
    masked_logits = logits.clone()
    masked_logits[:, ~mask] = -1e9
    return masked_logits


def evaluate_task(model: nn.Module, test_loader, task_id: int,
                  fine_to_superclass: np.ndarray, superclass_order: List[int],
                  eval_protocol: str, device: str = 'cuda') -> float:
    """
    评估单个任务的准确率
    
    Args:
        model: 神经网络模型
        test_loader: 测试数据加载器
        task_id: 要评估的任务ID
        fine_to_superclass: fine label到superclass的映射
        superclass_order: superclass的顺序
        eval_protocol: 评估协议 ('cil' 或 'til')
        device: 计算设备
    
    Returns:
        accuracy: 准确率 (%)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            
            if eval_protocol == 'til':
                # TIL模式: 应用掩码屏蔽
                logits = apply_task_mask(logits, task_id, fine_to_superclass, superclass_order)
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


def train_task(model: nn.Module, train_loader, test_loader,
               layer_reg: LayerWiseRegularizer = None,
               epochs: int = 20, lr: float = 5e-5,
               device: str = 'cuda',
               task_id: int = None, seen_class_mask: torch.Tensor = None) -> Tuple[float, float]:
    """
    训练一个task
    
    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        layer_reg: 分层正则化器
        epochs: 训练轮数
        lr: 学习率
        device: 计算设备
        task_id: 当前任务ID (用于正则化器)
        seen_class_mask: 已见类别的布尔掩码 (shape: [num_classes])
    
    Returns:
        (final_train_acc, final_test_acc)
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(features)
            
            # 训练时logit掩码：将未见过类别的logits设为-1e9
            # 这样softmax只在已见类别上计算，但labels仍使用原始索引
            if seen_class_mask is not None:
                masked_logits = logits.clone()
                masked_logits[:, ~seen_class_mask] = -1e9
                loss = F.cross_entropy(masked_logits, labels)
            else:
                loss = F.cross_entropy(logits, labels)

            # 添加正则化惩罚
            if layer_reg is not None:
                reg_loss = layer_reg.penalty(task_id=task_id, seen_class_mask=seen_class_mask)
                loss = loss + reg_loss

            loss.backward()

            # SI需要在线更新W
            if layer_reg is not None:
                if layer_reg.hidden_reg_type == 'si' or layer_reg.classifier_reg_type == 'si':
                    if layer_reg.hidden_regularizer is not None and isinstance(layer_reg.hidden_regularizer, SI):
                        layer_reg.hidden_regularizer.update_W()
                    if layer_reg.classifier_regularizer is not None and isinstance(layer_reg.classifier_regularizer, SI):
                        layer_reg.classifier_regularizer.update_W()

            optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total

    # 测试准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            
            # 测试时应用掩码（使用与训练相同的方式）
            if seen_class_mask is not None:
                logits = logits.clone()
                logits[:, ~seen_class_mask] = -1e9
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = 100. * correct / total

    return train_acc, test_acc


def run_continual_learning(hidden_reg: str, classifier_reg: str,
                           seed: int = 0, num_tasks: int = 20,
                           epochs: int = 20, device: str = 'cuda',
                           eval_protocol: str = 'cil',
                           hidden_dim: int = 1000,
                           arch: str = '2layer') -> Dict:
    """
    运行连续学习实验

    Args:
        hidden_reg: Hidden层正则化方法
        classifier_reg: Classifier层正则化方法
        seed: 随机种子
        num_tasks: task数量
        epochs: 每个task的训练轮数
        eval_protocol: 评估协议 ('cil' 或 'til')
        hidden_dim: Hidden层维度 (默认1000, 仅2层架构使用)
        arch: 架构类型 ('2layer' 或 '4layer')

    Returns:
        实验结果字典
    """
    method_name = f"{hidden_reg or 'none'}_{classifier_reg or 'none'}"
    print(f"\n{'='*70}")
    print(f"Method: Hidden={hidden_reg or 'None'}, Classifier={classifier_reg or 'None'}")
    print(f"Seed: {seed}, Tasks: {num_tasks}, Epochs: {epochs}")
    print(f"Eval Protocol: {eval_protocol.upper()}")
    print(f"Architecture: {arch}")
    if arch == '2layer':
        print(f"Hidden Dim: {hidden_dim}")
    print(f"{'='*70}")

    set_seed(seed)

    # 初始化数据
    cifar_data = SuperclassSplitCIFAR100(seed=seed, device=device)

    # 初始化模型 - 根据arch参数选择架构
    model = create_mlp_model(
        arch=arch,
        hidden_dim=hidden_dim,
        use_cosine_classifier=False
    ).to(device)

    # 初始化分层正则化器
    layer_reg = None
    if hidden_reg is not None or classifier_reg is not None:
        layer_reg = LayerWiseRegularizer(
            model,
            hidden_reg=hidden_reg,
            classifier_reg=classifier_reg,
            lambda_si=1.0,
            lambda_ewc=5000.0,
            lambda_mas=0.1,
            lambda_biocs=0.001
        )

    # 记录结果
    results = {
        'method': method_name,
        'hidden_reg': hidden_reg,
        'classifier_reg': classifier_reg,
        'seed': seed,
        'eval_protocol': eval_protocol,
        'arch': arch,
        'hidden_dim': hidden_dim if arch == '2layer' else None,
        'task_order': cifar_data.superclass_order,
        'R_matrix': np.zeros((num_tasks, num_tasks)),
        'AA_history': [],
    }

    start_time = time.time()

    for task_id in range(num_tasks):
        print(f"\nTask {task_id+1}/{num_tasks}")

        # 获取task数据
        train_loader, test_loader = cifar_data.get_task_data(task_id, batch_size=256)
        
        # 计算已见类别掩码（用于SSR动态切片和logit掩码）
        # 当前任务及之前所有任务的类别都是"已见"的
        num_classes = 100  # CIFAR-100
        seen_class_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
        for t in range(task_id + 1):
            target_superclass = cifar_data.superclass_order[t]
            # 将numpy数组转换为Tensor进行比较
            fine_to_super_tensor = torch.from_numpy(cifar_data.fine_to_superclass).to(device)
            task_classes = torch.where(fine_to_super_tensor == target_superclass)[0]
            seen_class_mask[task_classes] = True
        
        # 通知正则化器任务即将开始（保存SI参数快照）
        if layer_reg is not None:
            if layer_reg.hidden_reg_type == 'si' and layer_reg.hidden_regularizer is not None:
                if isinstance(layer_reg.hidden_regularizer, SI):
                    layer_reg.hidden_regularizer.begin_task()
            if layer_reg.classifier_reg_type == 'si' and layer_reg.classifier_regularizer is not None:
                if isinstance(layer_reg.classifier_regularizer, SI):
                    layer_reg.classifier_regularizer.begin_task()

        # 训练
        train_acc, test_acc = train_task(
            model, train_loader, test_loader,
            layer_reg=layer_reg,
            epochs=epochs, lr=5e-5, device=device,
            task_id=task_id, seen_class_mask=seen_class_mask
        )

        # 评估当前task（使用指定的eval_protocol）
        current_acc = evaluate_task(
            model, test_loader, task_id,
            cifar_data.fine_to_superclass, cifar_data.superclass_order,
            eval_protocol, device
        )
        results['R_matrix'][task_id, task_id] = current_acc

        # 评估所有已学习的task
        if task_id > 0:
            for prev_task in range(task_id):
                _, prev_test_loader = cifar_data.get_task_data(prev_task, batch_size=256)
                prev_acc = evaluate_task(
                    model, prev_test_loader, prev_task,
                    cifar_data.fine_to_superclass, cifar_data.superclass_order,
                    eval_protocol, device
                )
                results['R_matrix'][task_id, prev_task] = prev_acc

        # 计算Average Accuracy
        aa = np.mean(results['R_matrix'][task_id, :task_id+1])
        results['AA_history'].append(aa)

        print(f"  R_{{{task_id+1},{task_id+1}}}: {current_acc:.2f}%")
        print(f"  AA after task {task_id+1}: {aa:.2f}%")

        # 更新正则化器的重要性
        if layer_reg is not None:
            layer_reg.compute_importance(train_loader, device)
            # SI需要额外更新omega
            if hidden_reg == 'si' or classifier_reg == 'si':
                layer_reg.update_si()

    # 计算最终指标
    R = results['R_matrix']
    T = num_tasks
    
    # Final AA: 最后一个task后的平均准确率
    final_aa = results['AA_history'][-1]
    
    # Plasticity: 平均 R_{t,t} (t>=2，排除第一个task)
    plasticity = np.mean([R[t, t] for t in range(1, T)])
    
    # Backward Transfer (BWT): 学习新任务后对旧任务的负面影响
    # BWT = mean(R_{T,j} - R_{j,j}) for j < T
    bwt = np.mean([R[T-1, j] - R[j, j] for j in range(T-1)])
    
    # Task 1 Retention: 第一个任务的保留率
    task1_retention = R[T-1, 0]

    results['final_AA'] = final_aa
    results['plasticity'] = plasticity
    results['BWT'] = bwt
    results['task1_retention'] = task1_retention
    results['total_time'] = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"Final Results:")
    print(f"  Final AA: {final_aa:.2f}%")
    print(f"  Plasticity: {plasticity:.2f}%")
    print(f"  BWT: {bwt:.2f}%")
    print(f"  Task 1 Retention: {task1_retention:.2f}%")
    print(f"  Total time: {results['total_time']/60:.1f} minutes")
    print(f"{'='*70}")

    return results


def run_all_experiments(seed: int = 0, num_tasks: int = 20, epochs: int = 20,
                        eval_protocol: str = 'cil', hidden_dim: int = 1000,
                        arch: str = '2layer'):
    """运行所有9组消融实验"""

    # 9组实验配置
    experiments = [
        ('none', 'none', 'Baseline'),
        ('mas', 'mas', 'MAS (Full)'),
        ('ewc', 'ewc', 'EWC (Full)'),
        ('ewc', 'biocs', 'EWC + SSR'),
        ('mas', 'biocs', 'MAS + SSR'),
        ('si', 'biocs', 'SI + SSR'),
        ('si', 'si', 'SI (Full)'),
        ('none', 'biocs', 'SGD + SSR'),  # SGD+SSR (仅classifier)
        ('biocs', 'biocs', 'SSR (Full)'),  # 新增：SSR (Full) - hidden和classifier都使用SSR
    ]

    print("="*70)
    if arch == '2layer':
        print(f"2层MLP连续学习实验 - 9组消融实验 (hidden_dim={hidden_dim})")
    else:
        print("4层MLP连续学习实验 - 9组消融实验 (512->1000->512->256->100)")
    print("="*70)
    print(f"Configuration:")
    print(f"  Seed: {seed}")
    print(f"  Tasks: {num_tasks}")
    print(f"  Epochs per task: {epochs}")
    print(f"  Learning rate: 5e-5")
    print(f"  Eval Protocol: {eval_protocol.upper()}")
    print(f"  Architecture: {arch}")
    if arch == '2layer':
        print(f"  Hidden Dim: {hidden_dim}")
    print("="*70)

    all_results = {}

    for hidden_reg, classifier_reg, exp_name in experiments:
        results = run_continual_learning(
            hidden_reg=hidden_reg,
            classifier_reg=classifier_reg,
            seed=seed,
            num_tasks=num_tasks,
            epochs=epochs,
            device=device,
            eval_protocol=eval_protocol,
            hidden_dim=hidden_dim,
            arch=arch
        )
        all_results[exp_name] = results

    # 保存结果
    os.makedirs('../results', exist_ok=True)
    if arch == '2layer':
        output_file = f'../results/2layer_mlp_ablation_{eval_protocol}_h{hidden_dim}_seed{seed}.json'
    else:
        output_file = f'../results/4layer_mlp_ablation_{eval_protocol}_seed{seed}.json'

    # 转换numpy数组为列表以便JSON序列化
    for exp_name in all_results:
        all_results[exp_name]['R_matrix'] = all_results[exp_name]['R_matrix'].tolist()
        all_results[exp_name]['task_order'] = [int(x) for x in all_results[exp_name]['task_order']]

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("All experiments completed!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    # 打印汇总表格
    print("\n汇总结果:")
    print(f"{'Experiment':<20} {'Final AA':<12} {'Plasticity':<12} {'BWT':<12} {'Task1 Ret.':<12}")
    print("-"*70)
    for exp_name in ['Baseline', 'MAS (Full)', 'EWC (Full)', 'EWC + SSR',
                     'MAS + SSR', 'SI + SSR', 'SI (Full)', 'SGD + SSR', 'SSR (Full)']:
        if exp_name in all_results:
            r = all_results[exp_name]
            print(f"{exp_name:<20} {r['final_AA']:>10.2f}% {r['plasticity']:>10.2f}% {r['BWT']:>10.2f}% {r['task1_retention']:>10.2f}%")

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MLP连续学习实验 - 支持2层和4层架构')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--tasks', type=int, default=20, help='task数量')
    parser.add_argument('--epochs', type=int, default=20, help='每个task的训练轮数')
    parser.add_argument('--hidden', type=str, default=None, help='Hidden层正则化 (si/ewc/mas/biocs/none)')
    parser.add_argument('--classifier', type=str, default=None, help='Classifier层正则化 (si/ewc/mas/biocs/none)')
    parser.add_argument('--all', action='store_true', help='运行所有9组实验')
    parser.add_argument('--eval_protocol', type=str, default='cil', choices=['cil', 'til'],
                        help='评估协议: cil (Class-Incremental) 或 til (Task-Incremental)')
    parser.add_argument('--hidden_dim', type=int, default=1000, help='Hidden层维度 (默认1000, 仅2层架构使用)')
    parser.add_argument('--arch', type=str, default='2layer', choices=['2layer', '4layer'],
                        help='网络架构: 2layer (512->hidden_dim->100) 或 4layer (512->1000->512->256->100)')

    args = parser.parse_args()

    if args.all:
        run_all_experiments(seed=args.seed, num_tasks=args.tasks, epochs=args.epochs,
                           eval_protocol=args.eval_protocol, hidden_dim=args.hidden_dim,
                           arch=args.arch)
    elif args.hidden is not None or args.classifier is not None:
        results = run_continual_learning(
            hidden_reg=args.hidden if args.hidden != 'none' else None,
            classifier_reg=args.classifier if args.classifier != 'none' else None,
            seed=args.seed,
            num_tasks=args.tasks,
            epochs=args.epochs,
            device=device,
            eval_protocol=args.eval_protocol,
            hidden_dim=args.hidden_dim,
            arch=args.arch
        )
    else:
        print("请指定 --all 运行所有实验，或指定 --hidden 和 --classifier 运行单个实验")
        print("\n示例 (2层架构):")
        print("  python 2layer_mlp_study.py --all")
        print("  python 2layer_mlp_study.py --hidden ewc --classifier biocs --tasks 2 --epochs 5")
        print("  python 2layer_mlp_study.py --all --eval_protocol til  # TIL模式")
        print("  python 2layer_mlp_study.py --all --hidden_dim 256  # 使用256维hidden层")
        print("\n示例 (4层架构):")
        print("  python 2layer_mlp_study.py --all --arch 4layer")
        print("  python 2layer_mlp_study.py --all --arch 4layer --eval_protocol til")
        print("  python 2layer_mlp_study.py --hidden ewc --classifier biocs --arch 4layer")
