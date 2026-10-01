#!/usr/bin/env python3
"""
Elastic Weight Consolidation (EWC) 正则化实现
"""

import torch
import torch.nn as nn
from typing import Dict


class EWC:
    """
    Elastic Weight Consolidation - 使用Fisher信息矩阵保护重要参数
    
    核心思想:
    1. 计算Fisher信息矩阵估计参数重要性
    2. 对重要参数的变化施加更大惩罚
    """
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 5000.0):
        """
        Args:
            model: 要保护的模型
            lambda_ewc: EWC正则化强度
        """
        self.model = model
        self.lambda_ewc = lambda_ewc
        
        # 保存参考参数
        self.params: Dict[str, torch.Tensor] = {}
        # Fisher信息矩阵 (对角近似)
        self.fisher: Dict[str, torch.Tensor] = {}
        
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.params[n] = p.clone().detach()
                self.fisher[n] = torch.zeros_like(p)
    
    def compute_fisher(self, train_loader, device: str = 'cuda', task_id: int = None):
        """
        计算Fisher信息矩阵
        
        Args:
            train_loader: 训练数据加载器
            device: 计算设备
            task_id: 当前task的ID (TIL模式下需要)
        """
        self.model.eval()
        
        # 清零Fisher矩阵
        for n in self.fisher:
            self.fisher[n].zero_()
        
        # 累积梯度平方
        for batch in train_loader:
            # 处理不同长度的batch（可能是2或3个元素）
            if len(batch) == 2:
                features, labels = batch
            else:
                features, labels = batch[0], batch[1]
            features, labels = features.to(device), labels.to(device)
            self.model.zero_grad()
            
            # 根据模型类型选择forward方式
            if hasattr(self.model, 'use_til') and self.model.use_til:
                if task_id is None:
                    raise ValueError("TIL mode requires task_id for EWC compute_fisher")
                output = self.model(features, task_id=task_id)
            else:
                output = self.model(features)
            
            loss = torch.nn.functional.cross_entropy(output, labels)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.data ** 2
        
        # FIX: 按样本总数归一化（原实现按batch数，会被batch_size倍放大）
        total_samples = len(train_loader.dataset)
        for n in self.fisher:
            self.fisher[n] /= total_samples
    
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        计算EWC惩罚项
        
        Returns:
            EWC正则化损失
        """
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.params:
                loss += torch.sum(self.fisher[n] * (p - self.params[n]) ** 2)
        return self.lambda_ewc * loss
    
    def update_reference(self):
        """更新参考参数为当前参数值"""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.params[n] = p.clone().detach()


if __name__ == '__main__':
    # 测试EWC
    print("Testing EWC...")
    
    from models.two_layer_mlp import TwoLayerMLP
    
    model = TwoLayerMLP()
    ewc = EWC(model, lambda_ewc=5000.0)
    
    print(f"Number of protected parameters: {len(ewc.params)}")
    print("\n✓ EWC test passed!")
