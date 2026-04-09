# AI Glass Cleanliness Detection System

Sistema di Computer Vision per il rilevamento dello stato di pulizia delle superfici vetrate di facciate di edifici, progettato per immagini e video acquisiti da drone.

---

## Indice

1. [Panoramica del sistema](#1-panoramica-del-sistema)
2. [Architettura generale](#2-architettura-generale)
3. [Struttura del progetto](#3-struttura-del-progetto)
4. [Installazione](#4-installazione)
5. [Pipeline di generazione dati sintetici (Blender)](#5-pipeline-di-generazione-dati-sintetici-blender)
6. [Moduli dati (src/data)](#6-moduli-dati-srcdata)
7. [Modelli (src/models)](#7-modelli-srcmodels)
8. [Training (src/training)](#8-training-srctraining)
9. [Valutazione (src/evaluation)](#9-valutazione-srcevaluation)
10. [Inferenza (src/inference)](#10-inferenza-srcinference)
11. [Domain Adaptation (src/domain_adaptation)](#11-domain-adaptation-srcdomain_adaptation)
12. [Configurazioni (configs/)](#12-configurazioni-configs)
13. [Script di utilità (scripts/)](#13-script-di-utilità-scripts)
14. [Test (tests/)](#14-test-tests)
15. [Workflow end-to-end](#15-workflow-end-to-end)
16. [Metriche e gradi di pulizia](#16-metriche-e-gradi-di-pulizia)
17. [Stack tecnologico](#17-stack-tecnologico)

---

## 1. Panoramica del sistema

Il sistema riceve in input immagini o video di facciate di edifici riprese da drone e restituisce:

| Output | Descrizione |
|---|---|
| **Maschera vetro** | Maschera binaria pixel-level che identifica ogni superficie vetrata |
| **Heatmap sporco** | Mappa continua [0, 1] che quantifica il livello di sporcizia per ogni pixel di vetro |
| **Cleanliness Score** | Punteggio globale [0, 1] e grado da A (pulito) a F (molto sporco) |
| **Analisi per regione** | Punteggio e bounding box per ogni singola finestra |
| **Pannello visivo** | Immagine 4-colonne: originale | maschera vetro | heatmap sporco | score annotato |

Il sistema è addestrato interamente su **dati sintetici** generati con Blender, riducendo il gap sintetico–reale tramite un modulo dedicato di **domain adaptation**.

---

## 2. Architettura generale

```
┌─────────────────────────────────────────────────────────────────┐
│                     FASE 1 — DATI SINTETICI                     │
│                                                                 │
│  BuildingGenerator → MaterialManager → DirtSimulator            │
│        ↓                   ↓                ↓                   │
│  LightingRandomizer  CameraController  MaskExporter             │
│        └──────────────────┬─────────────────┘                   │
│                    RenderPipeline                               │
│                  (RGB + mask + dirt + JSON)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 2000+ scene sintetiche
┌──────────────────────────▼──────────────────────────────────────┐
│                     FASE 2 — TRAINING                           │
│                                                                 │
│  GlassSegmentationModel   DirtEstimationModel   MultitaskModel  │
│  (U-Net / DeepLabV3+)     (U-Net regressione)  (encoder shared) │
│                                                                 │
│  Loss: BCE+Dice+Focal     MSE+MAE+SSIM         Kendall (incert.)│
└──────────────────────────┬──────────────────────────────────────┘
                           │ checkpoint .pth
┌──────────────────────────▼──────────────────────────────────────┐
│                     FASE 3 — INFERENZA                          │
│                                                                 │
│  Predictor ──→ CleanlinessScore ──→ Visualizer                  │
│  (image / video / batch)                                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ opzionale
┌──────────────────────────▼──────────────────────────────────────┐
│               FASE 4 — DOMAIN ADAPTATION                        │
│                                                                 │
│  CycleGAN  →  PseudoLabeler  →  DomainAdaptationFinetuner       │
│  (syn→real)   (filtro conf.)   (encoder freeze + mixed batch)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Struttura del progetto

```
AI-CHALLENGE/
│
├── requirements.txt                  # Dipendenze Python
├── setup.py                          # Installazione pacchetto
├── .gitignore
├── pytest.ini                        # Configurazione test runner
│
├── configs/
│   ├── blender_config.yaml           # Parametri generazione Blender
│   ├── glass_seg_config.yaml         # Training segmentazione vetro
│   ├── dirt_est_config.yaml          # Training stima sporcizia
│   ├── multitask_config.yaml         # Training modello multi-task
│   └── domain_adaptation.yaml        # Pipeline domain adaptation
│
├── blender/                          # Eseguito dentro Blender (bpy)
│   ├── building_generator.py
│   ├── material_manager.py
│   ├── dirt_simulator.py
│   ├── lighting_randomizer.py
│   ├── camera_controller.py
│   ├── mask_exporter.py
│   ├── render_pipeline.py
│   └── run_generation.py             # Entry point CLI Blender
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── augmentations.py
│   │   └── dataloader.py
│   │
│   ├── models/
│   │   ├── glass_segmentation.py
│   │   ├── dirt_estimation.py
│   │   ├── multitask_model.py
│   │   └── losses.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── scheduler.py
│   │   ├── train_glass.py
│   │   ├── train_dirt.py
│   │   └── train_multitask.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── cleanliness_score.py
│   │   └── visualizer.py
│   │
│   ├── inference/
│   │   ├── predictor.py
│   │   ├── image_inference.py
│   │   └── video_inference.py
│   │
│   └── domain_adaptation/
│       ├── real_dataset.py
│       ├── domain_augmentations.py
│       ├── pseudo_labeling.py
│       ├── style_transfer.py
│       ├── finetuner.py
│       └── run_adaptation.py
│
├── scripts/
│   ├── generate_dataset.sh
│   ├── train_all.sh
│   └── evaluate.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_models.py
│   ├── test_metrics.py
│   ├── test_inference.py
│   └── test_domain_adaptation.py
│
└── dataset/
    ├── images/                       # RGB render (PNG)
    ├── glass_masks/                  # Maschere binarie vetro (PNG)
    ├── dirt_maps/                    # Heatmap sporcizia [0,255] (PNG)
    └── metadata/                     # JSON per scena (parametri, intrinsics)
```

---

## 4. Installazione

### Requisiti

- Python 3.10+
- CUDA 11.8+ (consigliato per training; CPU supportata per inferenza)
- Blender 4.x (solo per la generazione dati)

### Setup ambiente

```bash
# Clona o scarica il progetto
cd AI-CHALLENGE

# Crea ambiente virtuale
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Installa dipendenze
pip install -r requirements.txt

# Rendi il pacchetto importabile (modalità development)
pip install -e .
```

### Verifica installazione

```bash
pytest tests/ -v --tb=short
```

---

## 5. Pipeline di generazione dati sintetici (Blender)

Tutti i file in `blender/` vengono eseguiti **dentro** Blender tramite la sua API Python (`bpy`). Non richiedono un ambiente Python separato.

### Come funziona

Ogni **scena sintetica** produce 4 file:

| File | Contenuto |
|---|---|
| `scene_XXXX_rgb.png` | Render a colori (512×512, Cycles) |
| `scene_XXXX_glass_mask.png` | Maschera binaria vetro (bianco=vetro) |
| `scene_XXXX_dirt_map.png` | Heatmap sporcizia [0–255] |
| `scene_XXXX_meta.json` | Parametri scena: piani, finestre, sporcizia, camera intrinsics, lighting preset |

### Componenti

#### `building_generator.py` — `BuildingGenerator`
Genera facciate procedurali con BMesh. Configurable: numero di piani (3–15), finestre per piano, profondità telaio, dimensione pane. Ogni oggetto vetro viene taggato con `obj["is_glass"] = True` per i passi successivi.

#### `material_manager.py` — `MaterialManager`
Assegna materiali PBR tramite alberi di nodi Blender:
- **Vetro**: Principled BSDF con texture rumore per lo sporco (roughness modulata da DirtNoiseTexture → ColorRamp)
- **Facciata**: 5 preset — `concrete`, `brick`, `metal_panel`, `stone`, `frame`

#### `dirt_simulator.py` — `DirtSimulator`
Due funzioni principali:
1. `randomize_dirt()` — modifica i nodi Blender per variare visivamente lo sporco nel render
2. `generate_ground_truth_dirt_map()` — proietta lo sporco sullo spazio immagine via `world_to_camera_view` e produce la mappa NumPy ground-truth

5 pattern di sporco: `perlin`, `voronoi`, `streaks`, `dust_spots`, `water_stains`

#### `lighting_randomizer.py` — `LightingRandomizer`
Randomizza illuminazione con lo shader Nishita (cielo fisico): 5 preset meteo — `clear`, `overcast`, `golden_hour`, `harsh_noon`, `foggy`. Controlla sole, nebbia volumetrica e temperatura di colore.

#### `camera_controller.py` — `CameraController`
Simula il posizionamento drone: altitudine, distanza dalla facciata, offset laterale, pitch, roll, lunghezza focale. Esporta la matrice degli **intrinsics** (fx, fy, cx, cy) nel JSON di metadati.

#### `mask_exporter.py` — `MaskExporter`
Esporta le maschere con la tecnica del **material swap**: sostituisce temporaneamente tutti i materiali con emission flat (bianco=vetro, nero=resto) e renderizza con 1 sample (nessun rumore).

#### `render_pipeline.py` — `RenderPipeline`
Orchestratore dei 9 passi per scena: building → materiali → sporco → camera → lighting → render RGB → maschera vetro → dirt map → JSON. Ogni scena è riproducibile tramite seed.

#### `run_generation.py` — Entry point CLI

```bash
blender --background --python blender/run_generation.py -- \
    --config configs/blender_config.yaml \
    --start 0 \
    --end 2000 \
    --resume          # Salta scene già renderizzate
```

Oppure con lo script shell:

```bash
bash scripts/generate_dataset.sh
```

---

## 6. Moduli dati (src/data)

### `dataset.py`

Tre classi Dataset PyTorch, tutte con factory `from_config(config, split)`:

| Classe | Input | Output dict |
|---|---|---|
| `GlassSegmentationDataset` | RGB + maschera vetro | `image`, `mask`, `scene_id` |
| `DirtEstimationDataset` | RGB + dirt map + maschera vetro | `image`, `dirt_map`, `glass_mask`, `scene_id` |
| `MultitaskDataset` | RGB + entrambe le maschere | `image`, `glass_mask`, `dirt_map`, `scene_id` |

Il metodo `_discover_scene_ids()` interseca automaticamente le cartelle `images/`, `glass_masks/` e `dirt_maps/` per includere solo le scene complete. Lo split train/val/test è deterministico (seeded).

### `augmentations.py`

```python
from src.data.augmentations import build_transforms, build_multimask_transforms

# Per segmentazione (immagine + 1 maschera)
train_tfm, val_tfm = build_transforms(config)

# Per multitask (immagine + N maschere in lista)
train_tfm, val_tfm = build_multimask_transforms(config)
```

Pipeline di training include: flip orizzontale, ShiftScaleRotate, distorsione elastica/griglia, riflesso solare, nebbia, blur, sharpen, compressione JPEG. Pipeline di validation: solo resize + normalize ImageNet.

### `dataloader.py`

```python
from src.data.dataloader import build_multitask_dataloaders

train_loader, val_loader, test_loader = build_multitask_dataloaders(config)
```

Tutte le factory restituiscono la tripla `(train, val, test)` con `persistent_workers=True` quando `num_workers > 0`.

---

## 7. Modelli (src/models)

### `glass_segmentation.py` — `GlassSegmentationModel`

Wrapper attorno a [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch). Supporta 7 architetture: `unet`, `unetplusplus`, `deeplabv3plus`, `fpn`, `pan`, `manet`, `linknet`.

```python
from src.models.glass_segmentation import GlassSegmentationModel

model = GlassSegmentationModel.from_config(config)
# oppure direttamente:
model = GlassSegmentationModel(
    architecture="unet",
    encoder_name="resnet50",
    encoder_weights="imagenet",
)

logits = model(image_tensor)          # (B, 1, H, W) — logit grezzi
mask = torch.sigmoid(logits) > 0.5   # maschera binaria
```

Metodi utili:
- `freeze_encoder()` / `unfreeze_encoder()` — per fine-tuning progressivo
- `get_param_groups(encoder_lr, decoder_lr)` — learning rate differenziale

### `dirt_estimation.py` — `DirtEstimationModel`

U-Net con encoder EfficientNet-B4, output sigmoid (mappa continua [0,1]).

```python
from src.models.dirt_estimation import DirtEstimationModel

model = DirtEstimationModel.from_config(config)

# Input solo RGB
dirt_map = model(image_tensor)

# Input RGB + maschera vetro (4 canali — più preciso)
dirt_map = model(image_tensor, glass_mask_tensor)
```

### `multitask_model.py` — `MultitaskFacadeModel`

Encoder condiviso (SMP) + due decoder indipendenti (`seg_decoder`, `reg_decoder`), ognuno con blocchi DecoderBlock (upsample bilineare + double conv + skip connection) e testa separata.

```python
from src.models.multitask_model import MultitaskFacadeModel

model = MultitaskFacadeModel.from_config(config)
seg_logits, dirt_heatmap = model(image_tensor)
# seg_logits  : (B, 1, H, W) — logit segmentazione
# dirt_heatmap: (B, 1, H, W) — heatmap sporco [0, 1]
```

### `losses.py`

| Classe | Formula | Uso |
|---|---|---|
| `DiceLoss` | $1 - \frac{2\|p \cap g\|}{\|p\| + \|g\|}$ | Segmentazione sbilanciata |
| `FocalLoss` | $-\alpha(1-p)^\gamma \log p$ | Regioni difficili |
| `CombinedSegLoss` | BCE×0.4 + Dice×0.4 + Focal×0.2 | Default segmentazione |
| `SSIMLoss` | $1 - \text{SSIM}(p, g)$ | Regressione con struttura |
| `CombinedRegLoss` | MSE×0.5 + MAE×0.3 + SSIM×0.2 | Default regressione |
| `MultiTaskLoss` | Kendall 2018 — pesi incertezza learnable | Multi-task |

`MultiTaskLoss` impara automaticamente il bilanciamento tra segmentazione e regressione tramite parametri `log_var_seg` e `log_var_reg`:

$$\mathcal{L} = \frac{1}{2\sigma_s^2}\mathcal{L}_\text{seg} + \frac{1}{2\sigma_r^2}\mathcal{L}_\text{reg} + \log\sigma_s + \log\sigma_r$$

---

## 8. Training (src/training)

### `trainer.py` — `Trainer` (classe base)

Funzionalità incluse:
- **AMP** (Automatic Mixed Precision) con `torch.cuda.amp.GradScaler`
- **Gradient accumulation** (configurabile, default 4 step)
- **Early stopping** con patience configurabile
- **CheckpointManager** — salva i top-K checkpoint per metrica monitorata
- **TensorBoard** logging automatico di loss e metriche

### Addestrare il modello di segmentazione vetro

```bash
python -m src.training.train_glass \
    --config configs/glass_seg_config.yaml \
    --device cuda \
    --seed 42
```

### Addestrare il modello di stima sporcizia

```bash
python -m src.training.train_dirt \
    --config configs/dirt_est_config.yaml \
    --device cuda
```

### Addestrare il modello multi-task (consigliato)

```bash
python -m src.training.train_multitask \
    --config configs/multitask_config.yaml \
    --device cuda
```

### Addestrare tutti e tre in sequenza

```bash
bash scripts/train_all.sh
```

### Riprendere training da checkpoint

```bash
python -m src.training.train_multitask \
    --config configs/multitask_config.yaml \
    --resume checkpoints/multitask/best.pth
```

### `scheduler.py`

```python
from src.training.scheduler import build_optimizer, build_scheduler

optimizer = build_optimizer(model, config)   # AdamW con LR differenziale
scheduler = build_scheduler(optimizer, config)
# Tipi: cosine_warmup (default), plateau, step, cosine
```

---

## 9. Valutazione (src/evaluation)

### `metrics.py`

```python
from src.evaluation.metrics import segmentation_metrics, regression_metrics

# Segmentazione
metrics = segmentation_metrics(pred_mask, gt_mask, threshold=0.5)
# → {"iou": 0.82, "dice": 0.89, "precision": 0.91, "recall": 0.87, "f1": 0.89}

# Regressione (sporcizia)
metrics = regression_metrics(pred_dirt, gt_dirt, glass_mask)
# → {"mse": 0.012, "mae": 0.08, "rmse": 0.11, "psnr": 37.2, "ssim": 0.91}
```

### `cleanliness_score.py`

```python
from src.evaluation.cleanliness_score import compute_cleanliness_score, compute_full_analysis

# Punteggio globale
result = compute_cleanliness_score(glass_prob, dirt_heatmap)
print(result.overall_score)   # 0.73
print(result.grade)           # "B"

# Analisi per finestra (connected components)
result = compute_full_analysis(glass_prob, dirt_heatmap)
for region in result.per_region_scores:
    print(region)  # {"bbox": [x, y, w, h], "score": 0.68, "grade": "C"}
```

**Scala dei gradi:**

| Grado | Score | Significato |
|---|---|---|
| **A** | ≥ 0.85 | Pulito |
| **B** | ≥ 0.70 | Leggermente sporco |
| **C** | ≥ 0.50 | Moderatamente sporco |
| **D** | ≥ 0.30 | Molto sporco |
| **F** | < 0.30 | Critico |

### `visualizer.py`

```python
from src.evaluation.visualizer import make_result_panel, save_visualization

panel = make_result_panel(
    image,          # RGB numpy (H, W, 3)
    glass_prob,     # (H, W) float [0,1]
    dirt_heatmap,   # (H, W) float [0,1]
    result,         # CleanlinessResult
)
save_visualization(panel, "output/result.png")
```

Produce un pannello a 4 colonne: immagine originale | overlay maschera vetro (verde) | heatmap sporco (JET) | immagine annotata con grado.

### Valutazione completa su test set

```bash
python scripts/evaluate.py \
    --config configs/multitask_config.yaml \
    --checkpoint checkpoints/multitask/best.pth \
    --output results/
```

---

## 10. Inferenza (src/inference)

### `predictor.py` — `Predictor`

Interfaccia unificata che supporta 3 modalità:

| Modalità | Modelli caricati | Quando usare |
|---|---|---|
| `multitask` | MultitaskFacadeModel | Produzione (una sola forward pass) |
| `two_stage` | GlassSeg + DirtEst | Test comparativo |
| `glass_only` | GlassSegmentationModel | Solo rilevamento vetro |

```python
from src.inference.predictor import Predictor

# Carica da checkpoint salvati
predictor = Predictor.from_checkpoints(
    multitask_checkpoint="checkpoints/multitask/best.pth",
    device="cuda",
)

# Inferenza su immagine
result = predictor.predict_image("facade.jpg")
print(result["grade"])           # "B"
print(result["overall_score"])   # 0.73

# Inferenza su batch di array numpy
glass_probs, dirt_maps = predictor.predict_batch(image_list)
```

### Inferenza su immagine singola o cartella

```bash
# Immagine singola
python -m src.inference.image_inference \
    --checkpoint checkpoints/multitask/best.pth \
    --input facade.jpg \
    --output results/

# Cartella di immagini
python -m src.inference.image_inference \
    --checkpoint checkpoints/multitask/best.pth \
    --input-dir data/real_facades/ \
    --output results/
```

### Inferenza su video

```bash
python -m src.inference.video_inference \
    --checkpoint checkpoints/multitask/best.pth \
    --input drone_footage.mp4 \
    --output results/annotated.mp4
```

Caratteristiche video:
- `TemporalSmoother`: media mobile / EMA sui risultati per evitare jitter
- Esporta un JSON per frame con score e grade
- Produce video annotato con overlay grado in tempo reale

---

## 11. Domain Adaptation (src/domain_adaptation)

Modulo per ridurre il gap tra dati sintetici e immagini reali. Si usa **dopo** il training su dati sintetici.

### Pipeline completa

```bash
python -m src.domain_adaptation.run_adaptation \
    --config configs/domain_adaptation.yaml \
    --model-checkpoint checkpoints/multitask/best.pth \
    --stages style-transfer pseudo-label finetune
```

Le 3 fasi possono essere eseguite separatamente o in sequenza tramite `--stages`.

---

### Stage 1: Style Transfer (CycleGAN)

Addestra un CycleGAN per tradurre render sintetici → aspetto reale, senza coppie di immagini allineate.

**Architettura:**
- **Generator G** (syn→real): ResNet con 9 blocchi residui, padding riflessivo, Instance Norm
- **Generator F** (real→syn): stessa struttura
- **Discriminator D_A / D_B**: PatchGAN 70×70, 3 livelli, leaky ReLU
- **Loss**: LSGAN + cycle consistency (λ=10) + identity (λ=5)
- **ImagePool**: buffer da 50 immagini per stabilizzare il discriminatore

```bash
python -m src.domain_adaptation.run_adaptation \
    --config configs/domain_adaptation.yaml \
    --stages style-transfer
```

---

### Stage 2: Pseudo-Labeling

Usa il modello preaddestrato per generare maschere pseudo-label sulle immagini reali non etichettate. Accetta solo le predizioni ad alta confidenza.

**Filtri disponibili:**

| Strategia | Condizione di accettazione |
|---|---|
| `confidence` | `mean(max(p, 1-p)) ≥ 0.85` |
| `entropy` | `mean(H) ≤ 0.4` |
| `combined` (default) | entrambe le condizioni |

```bash
python -m src.domain_adaptation.run_adaptation \
    --config configs/domain_adaptation.yaml \
    --model-checkpoint checkpoints/multitask/best.pth \
    --stages pseudo-label
```

Output per ogni immagine accettata:
- `{stem}_glass.png` — maschera vetro binaria
- `{stem}_dirt.png` — heatmap sporco
- `{stem}_glass_conf.png` — mappa confidenza vetro
- `{stem}_dirt_conf.png` — mappa confidenza sporco
- `{stem}_meta.json` — statistiche
- `manifest.json` — lista completa accepted/rejected

---

### Stage 3: Fine-tuning

Fine-tuna il modello su un mix di dati reali (pseudo-labeled) + sintetici.

**Strategia:**
1. **Freeze encoder** per i primi N epoch (warm-up solo dei decoder)
2. **Unfreeze progressivo** dopo N epoch con LR encoder 100× più basso
3. **Batch misto**: 30% sintetico + 70% reale (ratio configurabile)
4. **Loss pesata per confidenza**: pixel a bassa confidenza contribuiscono meno

```bash
python -m src.domain_adaptation.run_adaptation \
    --config configs/domain_adaptation.yaml \
    --model-checkpoint checkpoints/multitask/best.pth \
    --stages finetune \
    --pseudo-manifest data/pseudo_labels/manifest.json
```

---

### Dataset classes

| Classe | Uso |
|---|---|
| `RealFacadeDataset` | Immagini reali, etichette opzionali |
| `PseudoLabeledDataset` | Immagini reali + pseudo-label + mappe confidenza |
| `MixedDataset` | Combina sintetico + reale con `WeightedRandomSampler` |

### Augmentations

```python
from src.domain_adaptation.domain_augmentations import (
    build_domain_randomization_transform,  # 15 aug pesanti per syn→real
    build_real_world_val_transform,        # solo resize+normalize
    build_synthetic_domain_randomization,  # aug leggero per sintetici
)
```

---

## 12. Configurazioni (configs/)

### `blender_config.yaml` — Generazione dati

```yaml
generation:
  n_scenes: 2000
  output_dir: dataset/
  render:
    resolution: [512, 512]
    samples: 128           # campioni Cycles (qualità vs velocità)
building:
  floors_range: [3, 15]
  windows_per_floor_range: [3, 8]
dirt:
  patterns: [perlin, voronoi, streaks, dust_spots, water_stains]
  intensity_range: [0.1, 0.9]
```

### `glass_seg_config.yaml` / `dirt_est_config.yaml` / `multitask_config.yaml`

```yaml
model:
  architecture: unet         # unet | unetplusplus | deeplabv3plus | fpn
  encoder_name: resnet50     # resnet50 | efficientnet-b4 | mit_b2
  encoder_weights: imagenet

training:
  n_epochs: 50
  batch_size: 16
  lr: 1e-3
  weight_decay: 1e-4
  scheduler: cosine_warmup
  warmup_epochs: 5
  amp: true                  # Mixed precision
  grad_accumulation_steps: 4
  early_stopping_patience: 10
```

### `domain_adaptation.yaml`

```yaml
paths:
  real_images_dir: data/real/images/
  pseudo_label_dir: data/pseudo_labels/
  cyclegan_checkpoint_dir: checkpoints/cyclegan/
  finetune_checkpoint_dir: checkpoints/finetuned/

pseudo_labeling:
  glass_confidence_threshold: 0.85
  max_mean_entropy: 0.4
  filter_strategy: combined   # confidence | entropy | combined
  max_pseudo_images: 1000

finetuning:
  mix_synthetic: true
  synthetic_ratio: 0.3
  freeze_encoder_epochs: 3
  n_epochs: 30
  lr: 5e-5
  encoder_lr_multiplier: 0.01
```

---

## 13. Script di utilità (scripts/)

| Script | Uso |
|---|---|
| `scripts/generate_dataset.sh` | Lancia la generazione Blender (2000 scene) |
| `scripts/train_all.sh` | Addestra glass seg → dirt est → multitask |
| `scripts/evaluate.py` | Valutazione completa su test set con report JSON |

```bash
# Generazione dataset
bash scripts/generate_dataset.sh

# Training sequenziale
bash scripts/train_all.sh

# Valutazione
python scripts/evaluate.py \
    --config configs/multitask_config.yaml \
    --checkpoint checkpoints/multitask/best.pth
```

---

## 14. Test (tests/)

```bash
# Tutti i test
pytest tests/ -v

# Solo un modulo
pytest tests/test_domain_adaptation.py -v

# Con coverage
pytest tests/ --cov=src --cov-report=html
```

| File | Cosa testa |
|---|---|
| `test_dataset.py` | Scene discovery, split riproducibili, shape output |
| `test_models.py` | Loss functions (Dice, Focal, SSIM, Kendall), forward pass modelli |
| `test_metrics.py` | IoU, Dice, MAE, PSNR, cleanliness score, gradi A–F |
| `test_inference.py` | Predictor con modelli mock, pre-processing, batch |
| `test_domain_adaptation.py` | Dataset reali, entropia, filtri pseudo-label, CycleGAN forward |

---

## 15. Workflow end-to-end

### Da zero a modello in produzione

```bash
# 1. Installa dipendenze
pip install -r requirements.txt
pip install -e .

# 2. Genera dataset sintetico (richiede Blender installato)
bash scripts/generate_dataset.sh
# → dataset/ con 2000 scene RGB + maschere + metadata

# 3. Addestra modello (scegli uno)
python -m src.training.train_multitask --config configs/multitask_config.yaml
# → checkpoints/multitask/best.pth

# 4. Valuta sul test set
python scripts/evaluate.py \
    --config configs/multitask_config.yaml \
    --checkpoint checkpoints/multitask/best.pth

# 5. Inferenza su immagine reale
python -m src.inference.image_inference \
    --checkpoint checkpoints/multitask/best.pth \
    --input facade.jpg \
    --output results/

# 6. (Opzionale) Domain adaptation su immagini reali non etichettate
python -m src.domain_adaptation.run_adaptation \
    --config configs/domain_adaptation.yaml \
    --model-checkpoint checkpoints/multitask/best.pth \
    --stages pseudo-label finetune
```

### Uso da codice Python

```python
from src.inference.predictor import Predictor
import cv2

# Carica modello
predictor = Predictor.from_checkpoints(
    multitask_checkpoint="checkpoints/multitask/best.pth"
)

# Analizza immagine
image = cv2.cvtColor(cv2.imread("facade.jpg"), cv2.COLOR_BGR2RGB)
result = predictor.predict_image(image)

print(f"Grado: {result['grade']}")          # "B"
print(f"Score: {result['overall_score']:.2f}")  # 0.73

# Visualizza risultato
from src.evaluation.visualizer import make_result_panel, save_visualization
from src.evaluation.cleanliness_score import compute_full_analysis

analysis = compute_full_analysis(result["glass_mask"], result["dirt_map"])
panel = make_result_panel(image, result["glass_mask"], result["dirt_map"], analysis)
save_visualization(panel, "output/result.png")
```

---

## 16. Metriche e gradi di pulizia

### Segmentazione vetro

$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|} \qquad \text{Dice} = \frac{2|P \cap G|}{|P| + |G|}$$

### Stima sporcizia

$$\text{MSE} = \frac{1}{N}\sum_i(p_i - g_i)^2 \qquad \text{SSIM}(p,g) \in [-1, 1]$$

### Cleanliness Score

$$\text{score} = 1 - \frac{\sum_{i \in \text{glass}} \text{dirt}_i}{\sum_{i \in \text{glass}} 1}$$

| Score | Grado | Interpretazione |
|---|---|---|
| [0.85 – 1.00] | **A** | Vetro pulito — nessun intervento necessario |
| [0.70 – 0.85) | **B** | Leggermente sporco — pulizia ordinaria |
| [0.50 – 0.70) | **C** | Moderatamente sporco — pulizia programmata |
| [0.30 – 0.50) | **D** | Molto sporco — pulizia urgente |
| [0.00 – 0.30) | **F** | Critico — pulizia immediata |

---

## 17. Stack tecnologico

| Libreria | Versione | Uso |
|---|---|---|
| PyTorch | 2.1+ | Framework DL, AMP, DataLoader |
| segmentation-models-pytorch | 0.3.3 | Architetture U-Net, DeepLabV3+, FPN |
| timm | 0.9.12 | Encoder backbone (ResNet, EfficientNet, MiT) |
| albumentations | 1.3.1 | Augmentazioni immagini + maschere |
| OpenCV | 4.8 | I/O immagini, connected components, visualizzazione |
| Blender (bpy) | 4.x | Generazione dati sintetici Cycles |
| einops | 0.7 | Operazioni tensor leggibili |
| PyYAML | 6.0 | Caricamento configurazioni |
| pytest | 7.4 | Test suite |
| TensorBoard | 2.14 | Monitoring training |