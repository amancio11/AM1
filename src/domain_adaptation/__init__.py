"""
src/domain_adaptation
=====================
Synthetic-to-real domain adaptation module.

Submodules
----------
real_dataset          Dataset classes for real/pseudo-labeled images
domain_augmentations  Aggressive domain-randomization augmentations
pseudo_labeling       Pseudo-label generation + quality filtering
style_transfer        CycleGAN unpaired image-to-image translation
finetuner             Fine-tuning pipeline on real-world data
run_adaptation        CLI orchestrator for the full pipeline
"""

from src.domain_adaptation.real_dataset import (
    RealFacadeDataset,
    PseudoLabeledDataset,
    MixedDataset,
)
from src.domain_adaptation.domain_augmentations import (
    build_domain_randomization_transform,
    build_real_world_val_transform,
    build_synthetic_domain_randomization,
)
from src.domain_adaptation.pseudo_labeling import PseudoLabeler, load_accepted_paths
from src.domain_adaptation.style_transfer import (
    ResNetGenerator,
    PatchGANDiscriminator,
    CycleGANTrainer,
    ImagePool,
)
from src.domain_adaptation.finetuner import DomainAdaptationFinetuner

__all__ = [
    "RealFacadeDataset",
    "PseudoLabeledDataset",
    "MixedDataset",
    "build_domain_randomization_transform",
    "build_real_world_val_transform",
    "build_synthetic_domain_randomization",
    "PseudoLabeler",
    "load_accepted_paths",
    "ResNetGenerator",
    "PatchGANDiscriminator",
    "CycleGANTrainer",
    "ImagePool",
    "DomainAdaptationFinetuner",
]
