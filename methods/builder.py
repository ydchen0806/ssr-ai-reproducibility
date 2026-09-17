"""Factory for building continual learning methods."""

import torch
import torch.nn as nn

from .base import BaseContinualLearner
from .bioreg import BioReg
from .ewc import EWC
from .si import SI
from .mas import MAS
from .lwf import LwF
from .der import DERPlusPlus
from .sgd_baseline import SGDBaseline
from .biocs_lora import BioCsLoRA
from .biocs_plus import BioCsPlus
from .knowledge_editing import KnowledgeEditingBioCS
from .er_ace import ERACE
from .gdumb import GDumb
from .cls_er import CLSER
from .geometry_controls import GeometryControls
from .kd_matched import KDMatched


METHOD_REGISTRY: dict[str, type[BaseContinualLearner]] = {
    "sgd": SGDBaseline,
    "finetune": SGDBaseline,
    "naive": SGDBaseline,
    "bioreg": BioReg,
    "biocs": BioReg,
    "spatial_biocs": BioReg,
    "ewc": EWC,
    "si": SI,
    "mas": MAS,
    "lwf": LwF,
    "der++": DERPlusPlus,
    "derpp": DERPlusPlus,
    "biocs_lora": BioCsLoRA,
    "biocs_plus": BioCsPlus,
    "biocs+": BioCsPlus,
    "knowledge_editing": KnowledgeEditingBioCS,
    "er_ace": ERACE,
    "erace": ERACE,
    "gdumb": GDumb,
    "cls_er": CLSER,
    "clser": CLSER,
    "geometry_controls": GeometryControls,
    "geometry_control": GeometryControls,
    "kd": KDMatched,
    "kd_ewc": KDMatched,
    "kd_mas": KDMatched,
    "kd_si": KDMatched,
    "kd_center": KDMatched,
    "kd_protodecor": KDMatched,
    "kd_spectral": KDMatched,
    "kd_ssr": KDMatched,
}


def build_method(
    config: dict, model: nn.Module, device: torch.device
) -> BaseContinualLearner:
    name = config["name"].lower()
    if name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method '{name}'. Available: {list(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[name](model=model, device=device, config=config)
