from __future__ import annotations

import argparse
import json

import pytest
import torch

from experiments.cub200_continual_benchmark import (
    ClassConditionedMaskDecoder,
    PrototypeCosineMaskDecoder,
    build_segmentation_decoder,
    segmentation_head_config,
    ssr_warmup_ramp_multiplier,
    write_result_record,
)
from experiments.multidataset_segmentation import SegmentationResult, result_record


def test_segmentation_decoder_factory_preserves_additive_default():
    decoder = build_segmentation_decoder(num_classes=3, hidden_dim=16)
    assert isinstance(decoder, ClassConditionedMaskDecoder)


def test_prototype_cosine_head_uses_class_prototypes_as_mask_logits():
    decoder = PrototypeCosineMaskDecoder(
        num_classes=3,
        hidden_dim=8,
        feature_dim=4,
        logit_scale=3.0,
    )
    with torch.no_grad():
        decoder.classifier.weight[0].fill_(1.0)
        decoder.classifier.weight[1].fill_(-1.0)
        decoder.class_bias.weight[0].fill_(0.25)
        decoder.class_bias.weight[1].fill_(-0.5)
        decoder.background_prototype.copy_(
            torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        )

    feature = torch.randn(1, 4, 5, 7)
    features = feature.expand(2, -1, -1, -1).clone()
    logits = decoder(features, torch.tensor([0, 1]), (9, 11))

    assert logits.shape == (2, 1, 9, 11)
    pixel_embeddings = torch.nn.functional.normalize(
        decoder.decoder(decoder.proj(features)), dim=1
    )
    foreground = torch.nn.functional.normalize(
        decoder.classifier(torch.tensor([0, 1])), dim=1
    ).unsqueeze(-1).unsqueeze(-1)
    background = torch.nn.functional.normalize(
        decoder.background_prototype, dim=0
    ).view(1, -1, 1, 1)
    expected = 3.0 * (
        (pixel_embeddings * foreground).sum(dim=1, keepdim=True)
        - (pixel_embeddings * background).sum(dim=1, keepdim=True)
    ) + decoder.class_bias(torch.tensor([0, 1])).view(-1, 1, 1, 1)
    expected = torch.nn.functional.interpolate(
        expected, size=(9, 11), mode="bilinear", align_corners=False
    )
    torch.testing.assert_close(logits, expected)

    logits.square().mean().backward()
    gradient = decoder.classifier.weight.grad
    assert gradient is not None
    assert gradient[0].abs().sum() > 0
    assert gradient[1].abs().sum() > 0
    assert gradient[2].abs().sum() == 0
    assert decoder.background_prototype.grad is not None
    assert decoder.background_prototype.grad.abs().sum() > 0
    assert decoder.class_bias.weight.grad is not None
    assert decoder.class_bias.weight.grad[:2].abs().sum() > 0
    assert decoder.class_bias.weight.grad[2].abs().sum() == 0
    assert decoder.logit_scale.grad is not None
    assert decoder.logit_scale.grad.abs() > 0


def test_prototype_head_configuration_is_recorded_in_result_schema(tmp_path):
    args = argparse.Namespace(
        method="task_ssr",
        seed=7,
        kernel_family="cauchy",
        a_exc=1.0,
        a_inh=0.8,
        sigma_exc=0.16,
        sigma_inh=0.45,
        dataset="oxford_iiit_pet",
        output_dir=tmp_path,
        image_size=160,
        selection_lock_sha256="development_screen",
        lambda_ssr=0.05,
        lambda_kd=0.5,
        segmentation_head="prototype_cosine",
        hidden_dim=192,
        prototype_logit_scale=12.0,
    )
    result = SegmentationResult(
        task="segmentation",
        method="task_ssr",
        seed=7,
        mean_iou=60.0,
        mean_dice=75.0,
        avg_forgetting_iou=1.0,
        effective_rank=4.0,
        mean_abs_offdiag_cosine=0.1,
        num_tasks=8,
        num_classes=37,
        initial_model_hash="a" * 64,
        task_schedule_hash="b" * 64,
        optimizer_steps=100,
    )
    record = result_record(
        args,
        result,
        dataset_manifest={
            "dataset_fingerprint": "dataset-v1",
            "mask_policy": "foreground-v1",
        },
        cache_manifest={
            "cache_sha256": "c" * 64,
            "encoder_state_sha256": "d" * 64,
            "torchvision_version": "test",
        },
        elapsed_s=1.0,
    )

    assert record["model"] == "resnet18_dense_prototype_cosine_decoder"
    assert record["model_config"] == segmentation_head_config(
        "prototype_cosine", 192, prototype_logit_scale=12.0
    )


def test_cub_result_record_distinguishes_prototype_head(tmp_path):
    args = argparse.Namespace(
        kernel_family="gaussian",
        a_exc=1.0,
        a_inh=0.8,
        sigma_exc=0.2,
        sigma_inh=0.5,
        segmentation_head="prototype_cosine",
        seg_hidden_dim=128,
        prototype_logit_scale=9.0,
        selection_lock_sha256="development_screen",
        lambda_sp=0.1,
    )
    row = {
        "task": "segmentation",
        "method": "biocs",
        "seed": 3,
        "mean_iou": 50.0,
        "mean_dice": 60.0,
        "avg_forgetting_iou": 2.0,
        "effective_rank": 4.0,
        "mean_abs_offdiag_cosine": 0.1,
        "initial_model_hash": "a" * 64,
        "task_schedule_hash": "b" * 64,
        "optimizer_steps": 10,
    }
    write_result_record(
        outdir=tmp_path,
        row=row,
        args=args,
        elapsed_s=1.0,
        dataset_hash="c" * 64,
    )
    path = tmp_path / "result_records/segmentation/biocs/seed_3/result_record.json"
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["model"] == "resnet18_dense_prototype_cosine_decoder"
    assert record["model_config"] == segmentation_head_config(
        "prototype_cosine", 128, prototype_logit_scale=9.0
    )


def test_prototype_cosine_head_rejects_nonpositive_scale():
    with pytest.raises(ValueError, match="must be positive"):
        PrototypeCosineMaskDecoder(3, hidden_dim=8, feature_dim=4, logit_scale=0.0)


def test_prototype_cosine_head_clamps_learnable_scale():
    decoder = PrototypeCosineMaskDecoder(
        2, hidden_dim=8, feature_dim=4, logit_scale=10.0
    )
    assert decoder.logit_scale.requires_grad
    with torch.no_grad():
        decoder.logit_scale.fill_(100.0)
    features = torch.randn(2, 4, 3, 3)
    labels = torch.tensor([0, 1])
    pixel_embeddings = torch.nn.functional.normalize(
        decoder.decoder(decoder.proj(features)), dim=1
    )
    foreground = torch.nn.functional.normalize(
        decoder.classifier(labels), dim=1
    ).unsqueeze(-1).unsqueeze(-1)
    background = torch.nn.functional.normalize(
        decoder.background_prototype, dim=0
    ).view(1, -1, 1, 1)
    expected = 30.0 * (
        (pixel_embeddings * foreground).sum(dim=1, keepdim=True)
        - (pixel_embeddings * background).sum(dim=1, keepdim=True)
    ) + decoder.class_bias(labels).view(-1, 1, 1, 1)
    torch.testing.assert_close(decoder(features, labels, (3, 3)), expected)


def test_ssr_warmup_ramp_multiplier_uses_per_task_budget():
    values = [
        ssr_warmup_ramp_multiplier(
            step,
            10,
            warmup_fraction=0.2,
            ramp_fraction=0.2,
        )
        for step in range(10)
    ]
    assert values == [0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert ssr_warmup_ramp_multiplier(
        0, 10, warmup_fraction=0.0, ramp_fraction=0.0
    ) == 1.0
