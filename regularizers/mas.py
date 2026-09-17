#!/usr/bin/env python3
"""
Memory Aware Synapses (MAS) 正则化实现
"""

import torch
import torch.nn as nn
from typing import Dict


class MAS:
    """
    Memory Aware Synapses - 使用输出变化估计参数重要性
    
    核心思想:
    1. 估计每个参数对模型输出的影响
    2. 对影响大的参数施加更大惩罚
    """
    
    def __init__(self, model: nn.Module, lambda_mas: float = 0.1):
        """
        Args:
            model: 要保护的模型
            lambda_mas: MAS正则化强度
        """
        self.model = model
        self.lambda_mas = lambda_mas
        
        # 保存参考参数
        self.params: Dict[str, torch.Tensor] = {}
        # 参数重要性 (omega)
        self.omega: Dict[str, torch.Tensor] = {}
        
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.params[n] = p.clone().detach()
                self.omega[n] = torch.zeros_like(p)
    
    def compute_omega(self, train_loader, device: str = 'cuda', task_id: int = None):
        """
        计算参数重要性omega并累加到历史omega
        
        修复：使用临时omega计算当前task的重要性，然后累加到历史omega，
        避免清零历史重要性，实现跨任务累积。
        
        Args:
            train_loader: 训练数据加载器
            device: 计算设备
            task_id: 当前task的ID (TIL模式下需要)
        """
        self.model.eval()
        
        # 创建临时omega用于当前task计算（不清零历史omega）
        temp_omega: Dict[str, torch.Tensor] = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                temp_omega[n] = torch.zeros_like(p)
        
        # 累积梯度绝对值到临时omega
        for batch in train_loader:
            # 处理不同长度的batch（可能是2或3个元素）
            if len(batch) == 2:
                features, _ = batch
            else:
                features = batch[0]
            features = features.to(device)
            self.model.zero_grad()
            
            # 根据模型类型选择forward方式
            if hasattr(self.model, 'use_til') and self.model.use_til:
                if task_id is None:
                    raise ValueError("TIL mode requires task_id for MAS compute_omega")
                output = self.model(features, task_id=task_id)
            else:
                output = self.model(features)
            
            # FIX: 使用mean而非sum，避免梯度量级与batch_size/output_dim耦合
            loss = torch.mean(output ** 2)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None and n in temp_omega:
                    temp_omega[n] += p.grad.data.abs()
        
        # FIX: 按样本总数归一化（与EWC修复一致）
        total_samples = len(train_loader.dataset)
        for n in temp_omega:
            temp_omega[n] /= total_samples
            self.omega[n] += temp_omega[n]  # 关键：累加而不是替换
    
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        计算MAS惩罚项
        
        Returns:
            MAS正则化损失
        """
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.params:
                loss += torch.sum(self.omega[n] * (p - self.params[n]) ** 2)
        return self.lambda_mas * loss
    
    def update_reference(self):
        """更新参考参数为当前参数值"""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.params[n] = p.clone().detach()


if __name__ == '__main__':
    # 测试MAS
    print("Testing MAS...")
    
    from models.two_layer_mlp import TwoLayerMLP
    
    model = TwoLayerMLP()
    mas = MAS(model, lambda_mas=0.1)
    
    print(f"Number of protected parameters: {len(mas.params)}")
    print("\n✓ MAS test passed!")
