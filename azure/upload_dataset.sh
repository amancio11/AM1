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
