#!/usr/bin/env python3
"""
分层正则化管理器
支持为不同层配置不同正则化方法
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from .si import SI
from .ewc import EWC
from .mas import MAS
from .biocs import BioCS, compute_bio_cs


class LayerWiseRegularizer:
    """
    分层正则化管理器
    
    允许为Hidden层和Classifier层配置不同的正则化方法
    """
    
    def __init__(self, model: nn.Module,
                 hidden_reg: Optional[str] = None,
                 classifier_reg: Optional[str] = None,
                 lambda_si: float = 1.0,
                 lambda_ewc: float = 5000.0,
                 lambda_mas: float = 0.1,
                 lambda_biocs: float = 0.001,
                 lambda_biocs_hidden: Optional[float] = None,
                 lambda_biocs_clf: Optional[float] = None):
        """
        Args:
            model: 2层MLP模型
            hidden_reg: Hidden层正则化方法 ('si', 'ewc', 'mas', 'biocs', None)
            classifier_reg: Classifier层正则化方法 ('si', 'ewc', 'mas', 'biocs', None)
            lambda_si: SI正则化强度
            lambda_ewc: EWC正则化强度
            lambda_mas: MAS正则化强度
            lambda_biocs: SSR正则化强度（向后兼容，用于两层）
            lambda_biocs_hidden: Hidden层SSR强度（独立设置，优先于lambda_biocs）
            lambda_biocs_clf: Classifier层SSR强度（独立设置，优先于lambda_biocs）

        Note:
            关于Loss标度失衡：Hidden层相似度矩阵(128x128)比Classifier层(20x20)大40倍，
            建议分别设置lambda_biocs_hidden和lambda_biocs_clf以避免惩罚值失衡。
        """
        self.model = model
        self.hidden_reg_type = hidden_reg
        self.classifier_reg_type = classifier_reg

        # 分离两层的参数
        self.hidden_params = []
        self.classifier_params = []

        for n, p in model.named_parameters():
            if 'fc1' in n:  # Hidden layer
                self.hidden_params.append((n, p))
            elif 'fc2' in n:  # Classifier layer
                self.classifier_params.append((n, p))

        # 确定每层的SSR lambda值（向前兼容：如果独立参数未设置，使用全局lambda_biocs）
        hidden_biocs_lambda = lambda_biocs_hidden if lambda_biocs_hidden is not None else lambda_biocs
        clf_biocs_lambda = lambda_biocs_clf if lambda_biocs_clf is not None else lambda_biocs

        # 创建正则化器
        self.hidden_regularizer = self._create_regularizer(
            hidden_reg, lambda_si, lambda_ewc, lambda_mas, hidden_biocs_lambda
        )
        self.classifier_regularizer = self._create_regularizer(
            classifier_reg, lambda_si, lambda_ewc, lambda_mas, clf_biocs_lambda
        )
    
    def _create_regularizer(self, reg_type: Optional[str], 
                           lambda_si: float, lambda_ewc: float,
                           lambda_mas: float, lambda_biocs: float):
        """创建指定类型的正则化器"""
        if reg_type is None or reg_type == 'none':
            return None
        elif reg_type == 'si':
            return SI(self.model, lambda_si)
        elif reg_type == 'ewc':
            return EWC(self.model, lambda_ewc)
        elif reg_type == 'mas':
            return MAS(self.model, lambda_mas)
        elif reg_type == 'biocs':
            return BioCS(lambda_biocs)
        else:
            raise ValueError(f"Unknown regularizer type: {reg_type}")
    
    def compute_importance(self, train_loader, device: str = 'cuda', task_id: int = None):
        """
        计算参数重要性
        
        Args:
            train_loader: 训练数据加载器
            device: 计算设备
            task_id: 当前task的ID (TIL模式下需要)
        """
        # 计算Hidden层重要性
        if self.hidden_regularizer is not None:
            if isinstance(self.hidden_regularizer, EWC):
                self.hidden_regularizer.compute_fisher(train_loader, device, task_id)
            elif isinstance(self.hidden_regularizer, MAS):
                self.hidden_regularizer.compute_omega(train_loader, device, task_id)
            elif isinstance(self.hidden_regularizer, SI):
                # SI需要在线更新，在task结束时调用update_omega
                pass
        
        # 计算Classifier层重要性
        if self.classifier_regularizer is not None:
            if isinstance(self.classifier_regularizer, EWC):
                self.classifier_regularizer.compute_fisher(train_loader, device, task_id)
            elif isinstance(self.classifier_regularizer, MAS):
                self.classifier_regularizer.compute_omega(train_loader, device, task_id)
            elif isinstance(self.classifier_regularizer, SI):
                pass
    
    def update_si(self):
        """更新SI的omega（在task结束时调用）"""
        if self.hidden_regularizer is not None and isinstance(self.hidden_regularizer, SI):
            self.hidden_regularizer.update_omega()
        if self.classifier_regularizer is not None and isinstance(self.classifier_regularizer, SI):
            self.classifier_regularizer.update_omega()
    
    def penalty(self, task_id: int = None, seen_class_mask: torch.Tensor = None) -> torch.Tensor:
        """
        计算总惩罚项

        Args:
            task_id: 当前task的ID (TIL模式下需要，用于SSR)
            seen_class_mask: 已见类别的布尔掩码 (shape: [num_classes])
                           用于SSR动态切片，防止未训练维度污染

        Returns:
            总正则化损失
        """
        total_loss = 0.0

        # Hidden层惩罚
        if self.hidden_regularizer is not None:
            if isinstance(self.hidden_regularizer, BioCS):
                # SSR应用于hidden层权重（每个神经元是一个权重向量）
                # 检查是否为4层MLP
                if hasattr(self.model, 'arch_type') and self.model.arch_type == '4layer':
                    # 4层MLP: 对fc1, fc2, fc3都施加SSR
                    total_loss += self.hidden_regularizer.penalty(self.model.fc1.weight)
                    total_loss += self.hidden_regularizer.penalty(self.model.fc2.weight)
                    total_loss += self.hidden_regularizer.penalty(self.model.fc3.weight)
                else:
                    # 2层MLP: 只对fc1施加SSR
                    # W1 shape: [hidden_dim, input_dim]，每行是一个神经元
                    hidden_weight = self.model.fc1.weight
                    total_loss += self.hidden_regularizer.penalty(hidden_weight)
            else:
                total_loss += self.hidden_regularizer.penalty(self.model)

        # Classifier层惩罚
        if self.classifier_regularizer is not None:
            if isinstance(self.classifier_regularizer, BioCS):
                # SSR只作用于classifier权重
                # 根据模型类型选择正确的权重
                if hasattr(self.model, 'use_til') and self.model.use_til:
                    # TIL模式: 使用对应task的分类头
                    if task_id is None:
                        raise ValueError("TIL mode requires task_id for SSR penalty")
                    classifier_weight = self.model.task_heads[task_id].weight
                elif hasattr(self.model, 'arch_type') and self.model.arch_type == '4layer':
                    # 4层MLP: classifier权重在self.classifier中
                    classifier_weight = self.model.classifier.weight
                else:
                    # 2层MLP CIL模式: 使用单头分类器fc2
                    classifier_weight = self.model.fc2.weight
                
                # 动态切片：只对已见类别施加SSR排斥力
                if seen_class_mask is not None:
                    classifier_weight = classifier_weight[seen_class_mask, :]
                
                total_loss += self.classifier_regularizer.penalty(classifier_weight)
            else:
                total_loss += self.classifier_regularizer.penalty(self.model)

        return total_loss
    
    def get_config(self) -> Dict[str, Optional[str]]:
        """获取配置信息"""
        return {
            'hidden_reg': self.hidden_reg_type,
            'classifier_reg': self.classifier_reg_type
        }


if __name__ == '__main__':
    # 测试分层正则化
    print("Testing LayerWiseRegularizer...")
    
    import sys
    sys.path.append('..')
    from models.two_layer_mlp import TwoLayerMLP
    
    model = TwoLayerMLP()
    
    # 测试 EWC + SSR 组合
    layer_reg = LayerWiseRegularizer(
        model,
        hidden_reg='ewc',
        classifier_reg='biocs'
    )
    
    print(f"Config: {layer_reg.get_config()}")
    print(f"Hidden parameters: {len(layer_reg.hidden_params)}")
    print(f"Classifier parameters: {len(layer_reg.classifier_params)}")
    
    print("\n✓ LayerWiseRegularizer test passed!")
