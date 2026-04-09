#!/usr/bin/env bash
# =============================================================================
# azure/run_generation.sh
# =============================================================================
# Avvia la pipeline di generazione immagini sintetiche Blender sulla VM Azure.
# Manda il processo in background con tmux: rimane attivo anche dopo
# disconnessione SSH.
#
# UTILIZZO (dalla VM):
#   bash ~/AI-CHALLENGE/azure/run_generation.sh [N_START] [N_END] [CONFIG]
#
# ESEMPI:
#   # 500 immagini di test (default)
#   bash azure/run_generation.sh
#
#   # Immagini 0-200 con config custom
#   bash azure/run_generation.sh 0 200 configs/blender_config.yaml
#
#   # Riprendi da dove si era interrotto
#   bash azure/run_generation.sh 150 500   # --resume è sempre attivo
# =============================================================================

set -euo pipefail

# ---- Parametri (con default) ------------------------------------------------
START="${1:-0}"
END="${2:-500}"
CONFIG="${3:-configs/blender_config.yaml}"
SESSION="blender_gen"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- Colori -----------------------------------------------------------------
GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}  Generazione immagini sintetiche — Blender${NC}"
echo -e "${CYAN}================================================${NC}"
echo "  Progetto : $PROJECT_DIR"
echo "  Config   : $CONFIG"
echo "  Range    : scene $START → $END  ($(( END - START )) immagini)"
echo ""

# ---- Verifica Blender -------------------------------------------------------
if ! command -v blender &>/dev/null; then
    echo -e "${YELLOW}  WARN: blender non trovato nel PATH${NC}"
    export PATH="$HOME/blender:$PATH"
    if ! command -v blender &>/dev/null; then
        echo "  ERROR: Blender non installato. Esegui prima setup_vm.sh"
        exit 1
    fi
fi
echo "  Blender  : $(blender --version 2>/dev/null | head -1)"

# ---- Crea cartelle output ---------------------------------------------------
mkdir -p "$PROJECT_DIR/dataset/images"
mkdir -p "$PROJECT_DIR/dataset/glass_masks"
mkdir -p "$PROJECT_DIR/dataset/dirt_maps"
mkdir -p "$PROJECT_DIR/dataset/metadata"
mkdir -p "$PROJECT_DIR/logs"

LOG_FILE="$PROJECT_DIR/logs/generation_$(date +%Y%m%d_%H%M%S).log"

# ---- Costruisci comando Blender ---------------------------------------------
BLENDER_CMD="blender --background --python $PROJECT_DIR/blender/run_generation.py \
    -- \
    --config $PROJECT_DIR/$CONFIG \
    --start $START \
    --end $END \
    --resume"

echo "  Log      : $LOG_FILE"
echo ""

# ---- Avvia in tmux (rimane vivo dopo disconnessione SSH) --------------------
if command -v tmux &>/dev/null; then
    # Chiudi sessione precedente se esiste
    tmux kill-session -t "$SESSION" 2>/dev/null || true

    tmux new-session -d -s "$SESSION" \
        "cd $PROJECT_DIR && $BLENDER_CMD 2>&1 | tee $LOG_FILE; echo 'GENERAZIONE COMPLETATA'; sleep 60"

    echo -e "${GREEN}  Processo avviato in sessione tmux: $SESSION${NC}"
    echo ""
    echo "  Comandi utili:"
    echo "    tmux attach -t $SESSION          # Entra nella sessione"
    echo "    Ctrl+B, D                        # Esci senza interrompere"
    echo "    tmux kill-session -t $SESSION    # Interrompi generazione"
    echo ""

else
    # Fallback senza tmux: esegui in foreground
    echo -e "${YELLOW}  tmux non disponibile — esecuzione in foreground${NC}"
    echo "  (La connessione SSH deve restare aperta)"
    echo ""
    cd "$PROJECT_DIR"
    eval "$BLENDER_CMD" 2>&1 | tee "$LOG_FILE"
    echo -e "${GREEN}  Generazione completata!${NC}"
fi

# ---- Funzione di monitoraggio -----------------------------------------------
cat << 'MONITOR'

  Per monitorare il progresso in un altro terminale SSH:

    # Numero immagini generate finora
    ls ~/AI-CHALLENGE/dataset/images/*.png 2>/dev/null | wc -l

    # Ultime righe del log
    tail -f ~/AI-CHALLENGE/logs/generation_*.log | tail -1

    # Spazio disco usato
    du -sh ~/AI-CHALLENGE/dataset/

MONITOR

# ---- Script di upload automatico su Azure Blob ------------------------------
UPLOAD_SCRIPT="$PROJECT_DIR/azure/upload_dataset.sh"
cat > "$UPLOAD_SCRIPT" << 'UPLOAD'
#!/usr/bin/env bash
# Carica il dataset generato su Azure Blob Storage
# Richiede: az login --use-device-code (o managed identity)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_ENV="$(dirname "${BASH_SOURCE[0]}")/vm_config.env"

if [ -f "$CONFIG_ENV" ]; then
    source "$CONFIG_ENV"
else
    # Fallback: leggi da variabili d'ambiente
    STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-}"
    STORAGE_KEY="${STORAGE_KEY:-}"
    STORAGE_CONTAINER="${STORAGE_CONTAINER:-dataset}"
fi

if [ -z "$STORAGE_ACCOUNT" ]; then
    echo "ERROR: STORAGE_ACCOUNT non impostato."
    echo "Imposta le variabili: export STORAGE_ACCOUNT=<nome> STORAGE_KEY=<chiave>"
    exit 1
fi

DATASET_DIR="$PROJECT_DIR/dataset"
echo "Upload $DATASET_DIR → az:///$STORAGE_ACCOUNT/$STORAGE_CONTAINER"

az storage blob upload-batch \
    --source "$DATASET_DIR" \
    --destination "$STORAGE_CONTAINER" \
    --account-name "$STORAGE_ACCOUNT" \
    --account-key "$STORAGE_KEY" \
    --overwrite true \
    --output table

echo "Upload completato!"
echo "Scarica in locale con: .\\azure\\sync_data.ps1 -Download"
UPLOAD
chmod +x "$UPLOAD_SCRIPT"
echo "  Script upload creato: azure/upload_dataset.sh"
