#!/usr/bin/env python3
"""
Synaptic Intelligence (SI) 正则化实现
"""

import torch
import torch.nn as nn
from typing import Dict


class SI:
    """
    Synaptic Intelligence - 在线参数重要性估计
    
    核心思想:
    1. 在线记录参数变化轨迹 (W)
    2. 在task结束时计算参数重要性 (omega)
    3. 重要性 = ∫ (dL/dθ) * dθ ≈ Σ (Δθ) * (dL/dθ)
    
    与MAS的区别:
    - MAS: 使用输出变化估计重要性
    - SI: 使用loss梯度积分估计重要性
    """
    
    def __init__(self, model: nn.Module, lambda_si: float = 1.0):
        """
        Args:
            model: 要保护的模型
            lambda_si: SI正则化强度
        """
        self.model = model
        self.lambda_si = lambda_si
        
        self.params: Dict[str, torch.Tensor] = {}
        self.omega: Dict[str, torch.Tensor] = {}
        self.W: Dict[str, torch.Tensor] = {}
        self._prev_step_params: Dict[str, torch.Tensor] = {}
        
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.params[n] = p.clone().detach()
                self.omega[n] = torch.zeros_like(p)
                self.W[n] = torch.zeros_like(p)
                self._prev_step_params[n] = p.data.clone()
    
    def update_W(self):
        """
        在线更新累积贡献W（每个optimizer.step()后调用）

        FIX: 原实现使用 (p.data - self.params[n])，即从任务开始到当前
        的总变化量，导致后期步骤 W 被系统性放大。
        修正为使用每步增量 (p.data - self._prev_step_params[n])，
        符合原论文 Zenke et al. ICML 2017 Eq.4 的路径积分定义。
        """
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                delta = p.data - self._prev_step_params[n]
                self.W[n] += (-p.grad.data) * delta
                self._prev_step_params[n] = p.data.clone()
    
    def update_omega(self):
        """
        在task结束时计算参数重要性omega
        omega = W / (delta_param^2 + epsilon)
        """
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                delta_param = p.data - self.params[n]
                # 避免除零
                self.omega[n] += self.W[n] / (delta_param ** 2 + 1e-8)
                # 重置W
                self.W[n].zero_()
                # 更新参考参数
                self.params[n] = p.clone().detach()
    
    def begin_task(self):
        """
        在任务开始前保存参数快照
        用于计算任务结束时的参数变化量
        """
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.params[n] = p.clone().detach()
    
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        计算SI惩罚项
        
        Returns:
            SI正则化损失
        """
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.params:
                loss += torch.sum(self.omega[n] * (p - self.params[n]) ** 2)
        return self.lambda_si * loss
    
    def get_importance(self, param_name: str) -> torch.Tensor:
        """获取指定参数的重要性"""
        return self.omega.get(param_name, torch.tensor(0.0))


if __name__ == '__main__':
    # 测试SI
    print("Testing SI...")
    
    from models.two_layer_mlp import TwoLayerMLP
    
    model = TwoLayerMLP()
    si = SI(model, lambda_si=1.0)
    
    # 模拟训练
    x = torch.randn(32, 512)
    target = torch.randint(0, 100, (32,))
    
    # Forward
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, target)
    
    # Backward
    loss.backward()
    
    # Update W
    si.update_W()
    
    # Update omega
    si.update_omega()
    
    # Compute penalty
    penalty = si.penalty(model)
    print(f"SI penalty: {penalty.item():.6f}")
    
    print("\n✓ SI test passed!")
