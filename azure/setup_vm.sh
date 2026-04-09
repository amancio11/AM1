#!/usr/bin/env bash
# =============================================================================
# azure/setup_vm.sh
# =============================================================================
# Eseguito SULLA VM Azure dopo il primo SSH.
#
# VM target: BOLOGNA-AI-AM-MACHINE
#   OS  : Ubuntu 24.04 LTS
#   GPU : AMD Radeon Instinct MI25 (Standard NV8as v4)
#   IP  : 72.146.184.240
#
# UTILIZZO — dalla VM (dopo aver caricato il progetto via Cloud Shell):
#   bash ~/AI-CHALLENGE/azure/setup_vm.sh
# =============================================================================

set -euo pipefail

# ---- Colori per log ---------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
step()  { echo -e "\n${CYAN}[$(date +%H:%M:%S)] $1${NC}"; }
ok()    { echo -e "${GREEN}  OK: $1${NC}"; }
warn()  { echo -e "${YELLOW}  WARN: $1${NC}"; }

# ---- Variabili --------------------------------------------------------------
BLENDER_VERSION="4.2.3"          # LTS release — supporta HIP per AMD GPU
BLENDER_ARCHIVE="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_URL="https://download.blender.org/release/Blender4.2/${BLENDER_ARCHIVE}"
BLENDER_INSTALL_DIR="$HOME/blender"

# Auto-detect project dir (supporta sia ~/AM1 che ~/AI-CHALLENGE)
if [ -d "$HOME/AM1" ]; then
    PROJECT_DIR="$HOME/AM1"
elif [ -d "$HOME/AI-CHALLENGE" ]; then
    PROJECT_DIR="$HOME/AI-CHALLENGE"
else
    PROJECT_DIR="$HOME/AI-CHALLENGE"  # default se nessuno esiste
fi

# Ubuntu 24.04 ha Python 3.12 come default; usiamo quello
PYTHON_BIN="python3"

# =============================================================================
# 1. AGGIORNAMENTO SISTEMA
# =============================================================================
step "Aggiorno pacchetti sistema (Ubuntu 24.04)"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -qq
ok "Sistema aggiornato"

# =============================================================================
# 2. DIPENDENZE SISTEMA
# =============================================================================
step "Installo dipendenze sistema"
sudo apt-get install -y -qq \
    build-essential \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    git \
    wget \
    curl \
    tar \
    unzip \
    htop \
    tmux \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libxi6 \
    libxkbcommon-x11-0 \
    libegl1 \
    libglu1-mesa \
    ffmpeg \
    pciutils
ok "Dipendenze di sistema installate"

# =============================================================================
# 3. GPU — DRIVER (NVIDIA CUDA o AMD ROCm)
# =============================================================================
step "Verifico GPU"

GPU_INFO=$(lspci | grep -i "vga\|display\|3d" || true)
echo "  GPU rilevata: $GPU_INFO"

GPU_TYPE="none"
if echo "$GPU_INFO" | grep -qi "nvidia"; then
    GPU_TYPE="nvidia"
elif echo "$GPU_INFO" | grep -qi "amd\|radeon\|advanced micro"; then
    GPU_TYPE="amd"
fi

if [ "$GPU_TYPE" = "nvidia" ]; then
    step "GPU NVIDIA rilevata — installo driver CUDA"
    sudo apt-get install -y -qq ubuntu-drivers-common
    sudo ubuntu-drivers install --gpgpu 2>/dev/null || sudo ubuntu-drivers autoinstall 2>/dev/null || true
    # Installa CUDA toolkit (per Blender + PyTorch)
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update -qq
    sudo apt-get install -y -qq cuda-toolkit-12-4 2>/dev/null || warn "cuda-toolkit non installato (riprova dopo reboot)"
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    ok "Driver NVIDIA installati — potrebbe servire reboot"
    warn "Dopo il setup esegui: sudo reboot"
elif [ "$GPU_TYPE" = "amd" ]; then
    step "GPU AMD rilevata — installo ROCm 6.x per Ubuntu 24.04"
    wget -q https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
        gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.1 noble main" | \
        sudo tee /etc/apt/sources.list.d/rocm.list
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.1/ubuntu noble main" | \
        sudo tee /etc/apt/sources.list.d/amdgpu.list
    sudo apt-get update -qq
    sudo apt-get install -y -qq rocm-hip-sdk rocm-opencl-runtime amdgpu-dkms
    echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo 'export HSA_OVERRIDE_GFX_VERSION=9.0.0' >> ~/.bashrc
    export PATH=/opt/rocm/bin:$PATH
    sudo usermod -aG render,video "$USER" 2>/dev/null || true
    ok "ROCm installato — potrebbe servire reboot"
    warn "Dopo il setup esegui: sudo reboot"
else
    warn "Nessuna GPU rilevata — Blender userà CPU (rendering ~6× più lento)"
fi

# =============================================================================
# 4. BLENDER 4.2 LTS (headless — supporto HIP AMD)
# =============================================================================
step "Scarico e installo Blender ${BLENDER_VERSION} LTS"

if [ -d "$BLENDER_INSTALL_DIR" ] && [ -f "$BLENDER_INSTALL_DIR/blender" ]; then
    warn "Blender già presente in $BLENDER_INSTALL_DIR, skip download"
else
    cd /tmp
    echo "  Scarico $BLENDER_URL..."
    wget -q --show-progress "$BLENDER_URL" -O "$BLENDER_ARCHIVE"
    mkdir -p "$BLENDER_INSTALL_DIR"
    tar -xf "$BLENDER_ARCHIVE" -C "$BLENDER_INSTALL_DIR" --strip-components=1
    rm "$BLENDER_ARCHIVE"
    ok "Blender estratto in $BLENDER_INSTALL_DIR"
fi

# Symlink globale
sudo ln -sf "$BLENDER_INSTALL_DIR/blender" /usr/local/bin/blender
blender --version 2>/dev/null | head -1 || warn "Blender installato ma --version ha avuto errori (normale headless)"
ok "Blender disponibile come: blender"

# =============================================================================
# 5. PROGETTO AI-CHALLENGE
# =============================================================================
step "Verifico progetto in $PROJECT_DIR"

if [ ! -d "$PROJECT_DIR" ]; then
    mkdir -p "$PROJECT_DIR"
    warn "Cartella progetto creata vuota."
    echo "  Carica i file con: bash ~/AI-CHALLENGE/azure/upload_via_cloudshell.sh"
else
    ok "Progetto trovato"
fi

# Crea cartelle necessarie
mkdir -p "$PROJECT_DIR/dataset/"{images,glass_masks,dirt_maps,metadata}
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/checkpoints"

# =============================================================================
# 6. AMBIENTE PYTHON VIRTUALE (Python 3.12 su Ubuntu 24.04)
# =============================================================================
step "Creo ambiente Python virtuale"

cd "$PROJECT_DIR"
if [ ! -d ".venv" ]; then
    $PYTHON_BIN -m venv .venv
    ok "Virtualenv creato (.venv)"
else
    warn "Virtualenv già esistente, skip"
fi

source .venv/bin/activate
pip install --upgrade pip setuptools wheel -q

# Installa PyTorch in base alla GPU rilevata
if lspci | grep -qi "nvidia"; then
    step "Installo PyTorch con supporto CUDA (GPU NVIDIA)"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q
    ok "PyTorch CUDA installato"
elif lspci | grep -qi "amd\|radeon"; then
    step "Installo PyTorch con supporto ROCm 6.1 (GPU AMD)"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1 -q
    ok "PyTorch ROCm installato"
else
    step "Installo PyTorch CPU-only"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
    ok "PyTorch CPU installato"
fi

# Installa dipendenze progetto
if [ -f "requirements.txt" ]; then
    step "Installo requirements.txt"
    pip install -r requirements.txt -q
    ok "Dipendenze Python installate"
else
    warn "requirements.txt non trovato — carica prima il progetto"
fi

# Pacchetto in modalità development
if [ -f "setup.py" ]; then
    pip install -e . -q
    ok "Pacchetto installato in modalità development"
fi

# =============================================================================
# 7. NUMPY NEL PYTHON INTERNO DI BLENDER
# =============================================================================
step "Installo numpy nel Python interno di Blender"

# Blender 4.2 usa Python 3.11
BLENDER_PY=$(find "$BLENDER_INSTALL_DIR" -name "python3*" -type f 2>/dev/null | grep bin | head -1 || true)
if [ -n "$BLENDER_PY" ]; then
    "$BLENDER_PY" -m ensurepip -q 2>/dev/null || true
    "$BLENDER_PY" -m pip install --upgrade pip -q 2>/dev/null || true
    "$BLENDER_PY" -m pip install numpy scipy Pillow PyYAML -q 2>/dev/null && ok "pacchetti installati in Blender Python" || warn "pip in Blender Python non disponibile (normale)"
fi

# =============================================================================
# 8. VERIFICA INSTALLAZIONE
# =============================================================================
step "Verifica installazione"

echo ""
echo "  Blender   : $(blender --version 2>/dev/null | head -1 || echo 'non disponibile')"
echo "  Python    : $($PYTHON_BIN --version)"
source "$PROJECT_DIR/.venv/bin/activate" 2>/dev/null || true
echo "  PyTorch   : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'non installato')"
echo "  GPU       : $(python -c 'import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print(\"CPU only\")' 2>/dev/null || echo 'non disponibile via Python (normale pre-reboot)')"
echo ""

# =============================================================================
# RIEPILOGO FINALE
# =============================================================================

echo "============================================================"
echo -e "${GREEN}  SETUP COMPLETATO!${NC}"
echo "============================================================"
echo ""
echo "  VM          : BOLOGNA-AI-AM-MACHINE (Ubuntu 24.04)"
echo "  GPU         : $(lspci | grep -i 'vga|display|3d' | head -1 || echo 'non rilevata')"
echo "  Progetto    : $PROJECT_DIR"
echo ""
echo "  SE HAI INSTALLATO driver GPU → esegui ora: sudo reboot"
echo "  Dopo il reboot riconnettiti e poi:"
echo ""
echo "  GENERA IMMAGINI SINTETICHE:"
echo "    bash $PROJECT_DIR/azure/run_generation.sh 0 500"
echo ""
echo "  MONITORA:"
echo "    tmux attach -t blender_gen"
echo "    ls $PROJECT_DIR/dataset/images/ | wc -l"
echo ""
echo "  SCARICA DATI (da Cloud Shell):"
echo "    bash $PROJECT_DIR/azure/upload_via_cloudshell.sh --download"
echo "============================================================"
