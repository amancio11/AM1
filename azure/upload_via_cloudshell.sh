#!/usr/bin/env bash
# =============================================================================
# azure/upload_via_cloudshell.sh
# =============================================================================
# Carica il progetto dalla tua macchina su Azure Cloud Shell,
# poi lo trasferisce sulla VM tramite scp.
#
# ESEGUITO in Azure Cloud Shell (shell.azure.com) — NON in locale.
#
# UTILIZZO:
#   # Trasferisci progetto sulla VM (dopo aver fatto upload da Cloud Shell)
#   bash upload_via_cloudshell.sh --upload
#
#   # Scarica dataset dalla VM in Cloud Shell (poi scarica da lì)
#   bash upload_via_cloudshell.sh --download
# =============================================================================

set -euo pipefail

# ---- Configurazione VM (dalla screenshot) -----------------------------------
VM_IP="72.146.184.240"
VM_USER="azureuser"                 # Cambia se hai scelto un altro username
SSH_KEY="$HOME/.ssh/bologna_vm"     # Chiave generata da Cloud Shell
PROJECT_REMOTE="/home/${VM_USER}/AI-CHALLENGE"
PROJECT_LOCAL="$HOME/AI-CHALLENGE"  # Path in Cloud Shell dopo l'upload

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
step() { echo -e "\n${CYAN}[$(date +%H:%M:%S)] $1${NC}"; }
ok()   { echo -e "${GREEN}  OK: $1${NC}"; }

MODE="${1:---upload}"

# =============================================================================
# GENERA CHIAVE SSH (solo prima volta)
# =============================================================================
generate_ssh_key() {
    if [ ! -f "$SSH_KEY" ]; then
        step "Genero chiave SSH in Cloud Shell"
        ssh-keygen -t ed25519 -f "$SSH_KEY" -N "" -C "cloudshell-bologna-vm"
        echo ""
        echo -e "${YELLOW}  ====================================================${NC}"
        echo -e "${YELLOW}  AZIONE RICHIESTA — aggiungi questa chiave pubblica alla VM:${NC}"
        echo ""
        echo "  1. Vai su portal.azure.com → BOLOGNA-AI-AM-MACHINE → 'Reset password'"
        echo "  2. Scegli 'Reset SSH public key'"
        echo "  3. Username: $VM_USER"
        echo "  4. Incolla questa chiave:"
        echo ""
        cat "$SSH_KEY.pub"
        echo ""
        echo -e "${YELLOW}  5. Clicca 'Update'"
        echo "  6. Poi riesegui questo script"
        echo -e "${YELLOW}  ====================================================${NC}"
        exit 0
    else
        ok "Chiave SSH già presente: $SSH_KEY"
    fi
}

# =============================================================================
# UPLOAD: Cloud Shell → VM
# =============================================================================
if [ "$MODE" = "--upload" ]; then
    generate_ssh_key

    step "Verifico connessione SSH alla VM ($VM_IP)"
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        "$VM_USER@$VM_IP" "echo 'Connessione OK'" || {
        echo "Errore SSH. Verifica che la chiave pubblica sia stata aggiunta alla VM."
        exit 1
    }
    ok "VM raggiungibile"

    step "Creo cartella progetto sulla VM"
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$VM_USER@$VM_IP" \
        "mkdir -p $PROJECT_REMOTE/{dataset/{images,glass_masks,dirt_maps,metadata},logs,checkpoints,azure}"

    step "Trasferisco progetto su VM (escludo dati pesanti)"
    # rsync è disponibile in Cloud Shell
    rsync -avz --progress \
        -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
        --exclude='.venv/' \
        --exclude='.git/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='dataset/images/' \
        --exclude='dataset/glass_masks/' \
        --exclude='dataset/dirt_maps/' \
        --exclude='checkpoints/*.pth' \
        --exclude='runs/' \
        --exclude='*.egg-info/' \
        "$PROJECT_LOCAL/" \
        "$VM_USER@$VM_IP:$PROJECT_REMOTE/"

    step "Rendo eseguibili gli script"
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$VM_USER@$VM_IP" \
        "chmod +x $PROJECT_REMOTE/azure/*.sh 2>/dev/null || true"

    ok "Upload completato!"
    echo ""
    echo "  PROSSIMO PASSO sulla VM:"
    echo "  ssh -i $SSH_KEY $VM_USER@$VM_IP"
    echo "  bash ~/AI-CHALLENGE/azure/setup_vm.sh"
fi

# =============================================================================
# DOWNLOAD: VM → Cloud Shell → Locale
# =============================================================================
if [ "$MODE" = "--download" ]; then
    generate_ssh_key

    step "Conto immagini disponibili sulla VM"
    COUNT=$(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$VM_USER@$VM_IP" \
        "ls $PROJECT_REMOTE/dataset/images/*.png 2>/dev/null | wc -l" || echo "0")
    echo "  Immagini generate: $COUNT"

    if [ "$COUNT" = "0" ]; then
        echo "  Nessuna immagine da scaricare ancora."
        exit 0
    fi

    DOWNLOAD_DIR="$HOME/dataset_download_$(date +%Y%m%d_%H%M)"
    mkdir -p "$DOWNLOAD_DIR"

    step "Scarico dataset dalla VM in Cloud Shell"
    rsync -avz --progress \
        -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
        "$VM_USER@$VM_IP:$PROJECT_REMOTE/dataset/" \
        "$DOWNLOAD_DIR/"

    ok "$COUNT immagini scaricate in: $DOWNLOAD_DIR"
    echo ""
    echo "  Per scaricare sul tuo PC:"
    echo "  1. In Cloud Shell, clicca l'icona Upload/Download (↑↓)"
    echo "  2. Scegli Download"
    echo "  3. Path: $DOWNLOAD_DIR"
    echo ""
    echo "  Oppure comprimi e scarica come ZIP:"
    echo "  cd $HOME && zip -r dataset_$(date +%Y%m%d).zip dataset_download_*/"
    echo "  Poi scarica dataset_$(date +%Y%m%d).zip da Cloud Shell"
fi
