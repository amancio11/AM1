"""
style_transfer.py
=================
CycleGAN-based unpaired image-to-image translation.

Translates synthetic renders (domain A) → real-looking images (domain B)
so that models trained on them generalize to actual drone footage.

Architecture
------------
Generator : ResNet-based (Johnson et al. 2016)
  - Reflection-padded convolutions to avoid border artefacts
  - Instance normalization (works at batch_size=1)
  - n_res_blocks=9 residual blocks (default)
  - Skip connections preserved via identity shortcut

Discriminator : 70×70 PatchGAN (Isola et al. 2017)
  - 3-layer leaky-ReLU convnet
  - Each output pixel classifies a 70×70 receptive-field patch
  - Instance normalization on all layers except first and last

Loss: LSGAN (least-squares) — more stable than BCE.

Training objective (CycleGAN):
  L_total = λ_adv * (L_GAN_G + L_GAN_F)
           + λ_cycle * (L_cycle_A + L_cycle_B)
           + λ_identity * (L_id_A + L_id_B)

ImagePool: 50-image replay buffer to stabilise discriminator training
  (Shrivastava et al. 2017).

Usage
-----
    from src.domain_adaptation.style_transfer import CycleGANTrainer

    trainer = CycleGANTrainer.from_config(config)
    trainer.train()

    # At inference time, convert a single synthetic image:
    stylized = trainer.translate_syn_to_real(syn_image_tensor)
"""

from __future__ import annotations

import copy
import functools
import itertools
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _norm_layer(norm_type: str = "instance") -> nn.Module:
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "batch":
        return functools.partial(nn.BatchNorm2d)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


class _ResBlock(nn.Module):
    """One residual block for the ResNet generator."""

    def __init__(self, channels: int, norm_layer):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            norm_layer(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            norm_layer(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ResNetGenerator(nn.Module):
    """
    ResNet generator for CycleGAN.

    Parameters
    ----------
    in_channels : int       (default 3)
    out_channels : int      (default 3)
    ngf : int               Number of generator filters in first conv layer.
    n_res_blocks : int      Number of residual blocks in bottleneck.
    norm_type : str         'instance' or 'batch'.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ngf: int = 64,
        n_res_blocks: int = 9,
        norm_type: str = "instance",
    ):
        super().__init__()
        NormLayer = _norm_layer(norm_type)

        # ------- Encoder (down-sampling) -------
        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, bias=False),
            NormLayer(ngf),
            nn.ReLU(inplace=True),
        ]
        curr_ch = ngf
        for _ in range(2):
            next_ch = curr_ch * 2
            encoder += [
                nn.Conv2d(curr_ch, next_ch, kernel_size=3, stride=2, padding=1, bias=False),
                NormLayer(next_ch),
                nn.ReLU(inplace=True),
            ]
            curr_ch = next_ch

        # ------- Bottleneck (residual blocks) -------
        bottleneck = [_ResBlock(curr_ch, NormLayer) for _ in range(n_res_blocks)]

        # ------- Decoder (up-sampling) -------
        decoder = []
        for _ in range(2):
            next_ch = curr_ch // 2
            decoder += [
                nn.ConvTranspose2d(
                    curr_ch, next_ch,
                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
                ),
                NormLayer(next_ch),
                nn.ReLU(inplace=True),
            ]
            curr_ch = next_ch

        decoder += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(curr_ch, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*encoder, *bottleneck, *decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.

    Parameters
    ----------
    in_channels : int
    ndf : int           Number of discriminator filters in first conv layer.
    n_layers : int      Number of strided conv blocks.
    norm_type : str
    """

    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm_type: str = "instance",
    ):
        super().__init__()
        NormLayer = _norm_layer(norm_type)

        # First layer — no norm
        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        curr_ch = ndf
        for i in range(1, n_layers):
            next_ch = min(curr_ch * 2, 512)
            stride = 2 if i < n_layers - 1 else 1
            layers += [
                nn.Conv2d(curr_ch, next_ch, kernel_size=4, stride=stride, padding=1, bias=False),
                NormLayer(next_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            curr_ch = next_ch

        # Output layer — 1 channel, no norm, no activation (LSGAN style)
        layers.append(
            nn.Conv2d(curr_ch, 1, kernel_size=4, stride=1, padding=1)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Image Pool (replay buffer)
# ---------------------------------------------------------------------------

class ImagePool:
    """
    50-image history pool for discriminator training.

    With probability 0.5, returns a random past generated image instead of
    the freshly generated one, stabilising D training (Shrivastava et al.).
    """

    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        self.images: List[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return images

        out = []
        for img in images.unbind(0):
            img = img.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img.clone())
                out.append(img)
            elif random.random() < 0.5:
                idx = random.randint(0, len(self.images) - 1)
                old = self.images[idx].clone()
                self.images[idx] = img.clone()
                out.append(old)
            else:
                out.append(img)
        return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# LSGAN losses
# ---------------------------------------------------------------------------

class LSGANLoss(nn.Module):
    """Least-squares GAN loss (Mao et al. 2017)."""

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


# ---------------------------------------------------------------------------
# CycleGAN Trainer
# ---------------------------------------------------------------------------

class CycleGANTrainer:
    """
    Orchestrates unpaired training of synthetic→real (G) and real→synthetic (F).

    Parameters
    ----------
    synthetic_loader : DataLoader  — domain A (synthetic renders)
    real_loader : DataLoader       — domain B (real images)
    checkpoint_dir : str
    ngf, ndf : int
    n_res_blocks : int
    n_d_layers : int
    lambda_cycle : float           — cycle consistency weight (default 10)
    lambda_identity : float        — identity loss weight (default 5)
    lr : float
    beta1 : float                  — Adam β₁ (default 0.5)
    n_epochs : int
    pool_size : int
    device : str
    """

    def __init__(
        self,
        synthetic_loader: DataLoader,
        real_loader: DataLoader,
        checkpoint_dir: str,
        ngf: int = 64,
        ndf: int = 64,
        n_res_blocks: int = 9,
        n_d_layers: int = 3,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
        lr: float = 2e-4,
        beta1: float = 0.5,
        n_epochs: int = 50,
        pool_size: int = 50,
        device: str = "cuda",
        save_every: int = 5,
    ):
        self.syn_loader = synthetic_loader
        self.real_loader = real_loader
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.n_epochs = n_epochs
        self.save_every = save_every
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # --- Models -----------------------------------------------------------
        self.G = ResNetGenerator(ngf=ngf, n_res_blocks=n_res_blocks).to(self.device)
        self.F = ResNetGenerator(ngf=ngf, n_res_blocks=n_res_blocks).to(self.device)
        self.D_A = PatchGANDiscriminator(ndf=ndf, n_layers=n_d_layers).to(self.device)
        self.D_B = PatchGANDiscriminator(ndf=ndf, n_layers=n_d_layers).to(self.device)

        # --- Losses -----------------------------------------------------------
        self.criterion_gan = LSGANLoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # --- Optimizers -------------------------------------------------------
        self.opt_G = optim.Adam(
            itertools.chain(self.G.parameters(), self.F.parameters()),
            lr=lr, betas=(beta1, 0.999),
        )
        self.opt_D_A = optim.Adam(self.D_A.parameters(), lr=lr, betas=(beta1, 0.999))
        self.opt_D_B = optim.Adam(self.D_B.parameters(), lr=lr, betas=(beta1, 0.999))

        # --- LR schedulers: linear decay from epoch 50% onward ---------------
        def _lr_lambda(epoch):
            n_half = n_epochs // 2
            return 1.0 - max(0, epoch - n_half) / max(n_half, 1)

        self.sch_G = optim.lr_scheduler.LambdaLR(self.opt_G, _lr_lambda)
        self.sch_D_A = optim.lr_scheduler.LambdaLR(self.opt_D_A, _lr_lambda)
        self.sch_D_B = optim.lr_scheduler.LambdaLR(self.opt_D_B, _lr_lambda)

        # --- Replay buffers ---------------------------------------------------
        self.pool_A = ImagePool(pool_size)
        self.pool_B = ImagePool(pool_size)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Dict,
        synthetic_loader: DataLoader,
        real_loader: DataLoader,
    ) -> "CycleGANTrainer":
        da_cfg = config.get("domain_adaptation", config)
        st_cfg = da_cfg.get("style_transfer", {})
        paths = da_cfg.get("paths", {})

        g_cfg = st_cfg.get("generator", {})
        d_cfg = st_cfg.get("discriminator", {})

        return cls(
            synthetic_loader=synthetic_loader,
            real_loader=real_loader,
            checkpoint_dir=paths.get("cyclegan_checkpoint_dir", "checkpoints/cyclegan"),
            ngf=g_cfg.get("ngf", 64),
            ndf=d_cfg.get("ndf", 64),
            n_res_blocks=g_cfg.get("n_res_blocks", 9),
            n_d_layers=d_cfg.get("n_layers", 3),
            lambda_cycle=st_cfg.get("lambda_cycle", 10.0),
            lambda_identity=st_cfg.get("lambda_identity", 5.0),
            lr=st_cfg.get("lr", 2e-4),
            beta1=st_cfg.get("beta1", 0.5),
            n_epochs=st_cfg.get("n_epochs", 50),
            pool_size=st_cfg.get("pool_size", 50),
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        for epoch in range(1, self.n_epochs + 1):
            loss_G_total = 0.0
            loss_D_total = 0.0
            n_batches = 0

            real_iter = iter(self.real_loader)
            for syn_batch in self.syn_loader:
                try:
                    real_batch = next(real_iter)
                except StopIteration:
                    real_iter = iter(self.real_loader)
                    real_batch = next(real_iter)

                real_A = self._extract_image(syn_batch).to(self.device)
                real_B = self._extract_image(real_batch).to(self.device)

                lG, lD = self._step(real_A, real_B)
                loss_G_total += lG
                loss_D_total += lD
                n_batches += 1

            n = max(n_batches, 1)
            logger.info(
                f"[CycleGAN] Epoch {epoch}/{self.n_epochs}  "
                f"G={loss_G_total/n:.4f}  D={loss_D_total/n:.4f}"
            )

            self.sch_G.step()
            self.sch_D_A.step()
            self.sch_D_B.step()

            if epoch % self.save_every == 0 or epoch == self.n_epochs:
                self.save_checkpoint(epoch)

    def _step(
        self, real_A: torch.Tensor, real_B: torch.Tensor
    ) -> Tuple[float, float]:
        """Single generator + discriminator update; returns (loss_G, loss_D)."""

        # ====== Generator update ======
        self.opt_G.zero_grad()

        # Identity
        idt_A = self.G(real_B)
        idt_B = self.F(real_A)
        loss_idt_A = self.criterion_identity(idt_A, real_B) * self.lambda_identity * 0.5
        loss_idt_B = self.criterion_identity(idt_B, real_A) * self.lambda_identity * 0.5

        # GAN losses
        fake_B = self.G(real_A)
        fake_A = self.F(real_B)
        loss_gan_G = self.criterion_gan(self.D_B(fake_B), is_real=True)
        loss_gan_F = self.criterion_gan(self.D_A(fake_A), is_real=True)

        # Cycle consistency
        rec_A = self.F(fake_B)
        rec_B = self.G(fake_A)
        loss_cyc_A = self.criterion_cycle(rec_A, real_A) * self.lambda_cycle
        loss_cyc_B = self.criterion_cycle(rec_B, real_B) * self.lambda_cycle

        loss_G = loss_gan_G + loss_gan_F + loss_cyc_A + loss_cyc_B + loss_idt_A + loss_idt_B
        loss_G.backward()
        self.opt_G.step()

        # ====== Discriminator D_A update ======
        self.opt_D_A.zero_grad()
        fake_A_pool = self.pool_A.query(fake_A.detach())
        loss_D_A = 0.5 * (
            self.criterion_gan(self.D_A(real_A), is_real=True)
            + self.criterion_gan(self.D_A(fake_A_pool), is_real=False)
        )
        loss_D_A.backward()
        self.opt_D_A.step()

        # ====== Discriminator D_B update ======
        self.opt_D_B.zero_grad()
        fake_B_pool = self.pool_B.query(fake_B.detach())
        loss_D_B = 0.5 * (
            self.criterion_gan(self.D_B(real_B), is_real=True)
            + self.criterion_gan(self.D_B(fake_B_pool), is_real=False)
        )
        loss_D_B.backward()
        self.opt_D_B.step()

        return float(loss_G.item()), float((loss_D_A + loss_D_B).item())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def translate_syn_to_real(self, image: torch.Tensor) -> torch.Tensor:
        """
        Convert a single synthetic image to the real domain.

        Parameters
        ----------
        image : (3, H, W) or (B, 3, H, W) tensor in [-1, 1]

        Returns
        -------
        Stylized image in [-1, 1].
        """
        self.G.eval()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return self.G(image.to(self.device))

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int) -> None:
        path = self.ckpt_dir / f"cyclegan_epoch{epoch:04d}.pth"
        torch.save({
            "epoch": epoch,
            "G": self.G.state_dict(),
            "F": self.F.state_dict(),
            "D_A": self.D_A.state_dict(),
            "D_B": self.D_B.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D_A": self.opt_D_A.state_dict(),
            "opt_D_B": self.opt_D_B.state_dict(),
        }, str(path))
        logger.info(f"Saved CycleGAN checkpoint → {path}")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.F.load_state_dict(ckpt["F"])
        self.D_A.load_state_dict(ckpt["D_A"])
        self.D_B.load_state_dict(ckpt["D_B"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D_A.load_state_dict(ckpt["opt_D_A"])
        self.opt_D_B.load_state_dict(ckpt["opt_D_B"])
        epoch = ckpt.get("epoch", 0)
        logger.info(f"Loaded CycleGAN checkpoint from epoch {epoch}")
        return epoch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_image(batch) -> torch.Tensor:
        """Handle both dict batches (from dataset) and plain tensors."""
        if isinstance(batch, dict):
            return batch["image"]
        return batch
