# =============================================================================
# azure/sync_data.ps1
# =============================================================================
# Sincronizza files tra macchina locale e VM Azure tramite SCP / Azure Blob.
#
# UTILIZZO:
#   # Carica il progetto sulla VM
#   .\azure\sync_data.ps1 -Upload
#
#   # Scarica solo il dataset generato (immagini + maschere)
#   .\azure\sync_data.ps1 -Download
#
#   # Carica E scarica in sequenza (sync completo)
#   .\azure\sync_data.ps1 -Upload -Download
#
#   # Mostra stato VM e spazio disco
#   .\azure\sync_data.ps1 -Status
#
#   # Spegni VM (stop billing compute, storage rimane minimo)
#   .\azure\sync_data.ps1 -Stop
#
#   # Riaccendi VM
#   .\azure\sync_data.ps1 -Start
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Upload,
    [switch]$Download,
    [switch]$Status,
    [switch]$Start,
    [switch]$Stop,
    [switch]$UploadDatasetToBlob   # Upload dataset su Blob Storage (alternativa a SCP)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# =============================================================================
# Carica configurazione generata da provision_vm.ps1
# =============================================================================

$CONFIG_FILE = "$PSScriptRoot\vm_config.env"

if (-not (Test-Path $CONFIG_FILE)) {
    Write-Error "File di configurazione non trovato: $CONFIG_FILE`nEsegui prima: .\azure\provision_vm.ps1"
    exit 1
}

# Parse del file .env
$config = @{}
Get-Content $CONFIG_FILE | Where-Object { $_ -notmatch '^\s*#' -and $_ -match '=' } | ForEach-Object {
    $key, $value = $_ -split '=', 2
    $config[$key.Trim()] = $value.Trim()
}

$VM_IP        = $config["VM_PUBLIC_IP"]
$ADMIN_USER   = $config["ADMIN_USER"]
$SSH_KEY      = $config["SSH_KEY_PATH"]
$RG           = $config["RESOURCE_GROUP"]
$VM_NAME      = $config["VM_NAME"]
$STORAGE_ACC  = $config["STORAGE_ACCOUNT"]
$STORAGE_KEY  = $config["STORAGE_KEY"]
$CONTAINER    = $config["STORAGE_CONTAINER"]

$PROJECT_LOCAL  = (Get-Location).Path       # Cartella locale AI-CHALLENGE
$PROJECT_REMOTE = "/home/$ADMIN_USER/AI-CHALLENGE"
$SSH_OPTS       = "-i `"$SSH_KEY`" -o StrictHostKeyChecking=no -o ConnectTimeout=15"

# =============================================================================
# Helper functions
# =============================================================================

function Write-Step([string]$msg) {
    Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] $msg" -ForegroundColor Cyan
}

function Invoke-SSH([string]$cmd) {
    $full = "ssh $SSH_OPTS $ADMIN_USER@$VM_IP `"$cmd`""
    Invoke-Expression $full
}

function Get-CurrentIP {
    # L'IP statico non cambia, ma per VM deallocate-riavviate potrebbe essere utile
    return (az network public-ip show `
        --resource-group $RG `
        --name "$VM_NAME-ip" `
        --query "ipAddress" -o tsv 2>$null)
}

# =============================================================================
# -Status : mostra stato VM e statistiche dataset
# =============================================================================

if ($Status) {
    Write-Step "Stato VM: $VM_NAME"
    $vmState = az vm get-instance-view `
        --resource-group $RG `
        --name $VM_NAME `
        --query "instanceView.statuses[1].displayStatus" -o tsv 2>$null
    Write-Host "  Stato  : $vmState"
    Write-Host "  IP     : $VM_IP"

    if ($vmState -eq "VM running") {
        Write-Step "Statistiche dataset sulla VM"
        Invoke-SSH "echo 'Immagini RGB    :' && ls $PROJECT_REMOTE/dataset/images/*.png 2>/dev/null | wc -l; echo 'Glass masks     :' && ls $PROJECT_REMOTE/dataset/glass_masks/*.png 2>/dev/null | wc -l; echo 'Dirt maps       :' && ls $PROJECT_REMOTE/dataset/dirt_maps/*.png 2>/dev/null | wc -l; echo 'Spazio usato    :' && du -sh $PROJECT_REMOTE/dataset/ 2>/dev/null"

        Write-Step "Log generazione (ultime 5 righe)"
        Invoke-SSH "ls -t $PROJECT_REMOTE/logs/generation_*.log 2>/dev/null | head -1 | xargs -r tail -5"
    }
    exit 0
}

# =============================================================================
# -Start : avvia VM
# =============================================================================

if ($Start) {
    Write-Step "Avvio VM: $VM_NAME"
    az vm start --resource-group $RG --name $VM_NAME
    # Aggiorna IP (statico, ma riconferma)
    $newIP = Get-CurrentIP
    Write-Host "  VM avviata. IP: $newIP" -ForegroundColor Green
    Write-Host "  SSH: ssh -i $SSH_KEY $ADMIN_USER@$newIP"
    exit 0
}

# =============================================================================
# -Stop : dealloca VM (ferma billing CPU/GPU)
# =============================================================================

if ($Stop) {
    Write-Step "ATTENZIONE: la VM verrà deallocata (nessun costo compute)"
    Write-Host "  Il disco e lo storage account vengono mantenuti." -ForegroundColor Yellow
    $confirm = Read-Host "  Confermi? [s/N]"
    if ($confirm -match '^[sS]$') {
        az vm deallocate --resource-group $RG --name $VM_NAME --no-wait
        Write-Host "  VM in deallocazione (asincrono)" -ForegroundColor Green
    } else {
        Write-Host "  Annullato."
    }
    exit 0
}

# =============================================================================
# -Upload : carica progetto sulla VM tramite rsync / scp
# =============================================================================

if ($Upload) {
    Write-Step "Upload progetto → VM ($VM_IP)"

    # File e cartelle da escludere (dati pesanti, ambienti virtuali, cache)
    $EXCLUDE_PATTERNS = @(
        "--exclude=.venv/"
        "--exclude=.git/"
        "--exclude=__pycache__/"
        "--exclude=*.pyc"
        "--exclude=dataset/images/"
        "--exclude=dataset/glass_masks/"
        "--exclude=dataset/dirt_maps/"
        "--exclude=checkpoints/"
        "--exclude=runs/"
        "--exclude=*.pth"
        "--exclude=*.egg-info/"
    )

    # Usa rsync se disponibile (molto più veloce per update incrementali)
    $rsyncAvail = Get-Command rsync -ErrorAction SilentlyContinue
    if ($rsyncAvail) {
        $excludeStr = $EXCLUDE_PATTERNS -join " "
        $rsyncCmd = "rsync -avz --progress -e `"ssh $SSH_OPTS`" $excludeStr `"$PROJECT_LOCAL/`" `"$ADMIN_USER@${VM_IP}:$PROJECT_REMOTE/`""
        Write-Host "  Usando rsync (incrementale)"
        Invoke-Expression $rsyncCmd
    } else {
        # Fallback: crea archivio e trasferisce con scp
        Write-Host "  rsync non disponibile — uso scp (più lento)"
        $tmpArchive = "$env:TEMP\ai_challenge_$(Get-Random).tar.gz"

        # Crea lista file da escludere per tar
        $excludeForTar = @(".venv", ".git", "__pycache__", "dataset/images", "dataset/glass_masks", "dataset/dirt_maps", "checkpoints", "runs") |
            ForEach-Object { "--exclude=./$_" }

        # Usa tar di Git Bash se disponibile
        $tarCmd = Get-Command tar -ErrorAction SilentlyContinue
        if ($tarCmd) {
            $excludeStr = ($excludeForTar -join " ")
            tar czf "$tmpArchive" $excludeForTar -C "$PROJECT_LOCAL" .
            scp -i "$SSH_KEY" -o StrictHostKeyChecking=no "$tmpArchive" "${ADMIN_USER}@${VM_IP}:/tmp/project.tar.gz"
            Invoke-SSH "mkdir -p $PROJECT_REMOTE && tar xzf /tmp/project.tar.gz -C $PROJECT_REMOTE && rm /tmp/project.tar.gz"
            Remove-Item $tmpArchive
        } else {
            Write-Error "Né rsync né tar disponibili. Installa Git for Windows (include tar e rsync)."
            exit 1
        }
    }

    # Copia anche vm_config.env sulla VM (per upload_dataset.sh)
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no "$CONFIG_FILE" "${ADMIN_USER}@${VM_IP}:${PROJECT_REMOTE}/azure/vm_config.env"

    # Rendi eseguibili gli script shell
    Invoke-SSH "chmod +x $PROJECT_REMOTE/azure/setup_vm.sh $PROJECT_REMOTE/azure/run_generation.sh 2>/dev/null || true"

    Write-Host ""
    Write-Host "  Upload completato!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  PROSSIMI PASSI sulla VM:"
    Write-Host "  1. ssh -i $SSH_KEY $ADMIN_USER@$VM_IP"
    Write-Host "  2. bash ~/AI-CHALLENGE/azure/setup_vm.sh      (solo prima volta)"
    Write-Host "  3. bash ~/AI-CHALLENGE/azure/run_generation.sh 0 500"
}

# =============================================================================
# -Download : scarica dataset generato dalla VM
# =============================================================================

if ($Download) {
    Write-Step "Download dataset dalla VM ($VM_IP) → locale"

    $LOCAL_DATASET = "$PROJECT_LOCAL\dataset"
    New-Item -ItemType Directory -Force -Path "$LOCAL_DATASET\images"     | Out-Null
    New-Item -ItemType Directory -Force -Path "$LOCAL_DATASET\glass_masks"| Out-Null
    New-Item -ItemType Directory -Force -Path "$LOCAL_DATASET\dirt_maps"  | Out-Null
    New-Item -ItemType Directory -Force -Path "$LOCAL_DATASET\metadata"   | Out-Null

    # Controlla quante immagini ci sono sulla VM
    $remoteCount = Invoke-SSH "ls $PROJECT_REMOTE/dataset/images/*.png 2>/dev/null | wc -l"
    Write-Host "  Immagini disponibili sulla VM: $remoteCount"

    if ([int]$remoteCount -eq 0) {
        Write-Host "  Nessuna immagine da scaricare." -ForegroundColor Yellow
        exit 0
    }

    $rsyncAvail = Get-Command rsync -ErrorAction SilentlyContinue
    if ($rsyncAvail) {
        # Download incrementale — scarica solo file nuovi
        foreach ($folder in @("images", "glass_masks", "dirt_maps", "metadata")) {
            Write-Host "  Scarico $folder..."
            $remotePath = "$ADMIN_USER@${VM_IP}:$PROJECT_REMOTE/dataset/$folder/"
            $localPath  = "$LOCAL_DATASET\$folder\"
            rsync -avz --progress -e "ssh $SSH_OPTS" "$remotePath" "$localPath"
        }
    } else {
        # Fallback scp
        Write-Host "  Uso scp (può essere lento per molti file)..."
        foreach ($folder in @("images", "glass_masks", "dirt_maps", "metadata")) {
            Write-Host "  Scarico $folder..."
            scp -r -i "$SSH_KEY" -o StrictHostKeyChecking=no `
                "${ADMIN_USER}@${VM_IP}:${PROJECT_REMOTE}/dataset/${folder}/*" `
                "$LOCAL_DATASET\$folder\"
        }
    }

    # Conta file scaricati
    $downloadedCount = (Get-ChildItem "$LOCAL_DATASET\images\*.png" -ErrorAction SilentlyContinue).Count
    Write-Host ""
    Write-Host "  Download completato: $downloadedCount immagini in $LOCAL_DATASET" -ForegroundColor Green
}

# =============================================================================
# -UploadDatasetToBlob : carica dataset su Azure Blob (alternativa al download)
# =============================================================================

if ($UploadDatasetToBlob) {
    Write-Step "Upload dataset dalla VM su Azure Blob Storage"
    Invoke-SSH "bash $PROJECT_REMOTE/azure/upload_dataset.sh"

    Write-Step "Download da Blob Storage in locale"
    $LOCAL_DATASET = "$PROJECT_LOCAL\dataset"
    New-Item -ItemType Directory -Force -Path $LOCAL_DATASET | Out-Null

    az storage blob download-batch `
        --source $CONTAINER `
        --destination $LOCAL_DATASET `
        --account-name $STORAGE_ACC `
        --account-key $STORAGE_KEY `
        --output table

    $count = (Get-ChildItem "$LOCAL_DATASET\images\*.png" -ErrorAction SilentlyContinue).Count
    Write-Host "  $count immagini scaricate in $LOCAL_DATASET" -ForegroundColor Green
}

# ---- Messaggio help se non è stato passato nessun parametro ----------------
if (-not ($Upload -or $Download -or $Status -or $Start -or $Stop -or $UploadDatasetToBlob)) {
    Write-Host ""
    Write-Host "Utilizzo:"
    Write-Host "  .\azure\sync_data.ps1 -Status               # Mostra stato VM"
    Write-Host "  .\azure\sync_data.ps1 -Upload               # Carica progetto sulla VM"
    Write-Host "  .\azure\sync_data.ps1 -Download             # Scarica dataset dalla VM"
    Write-Host "  .\azure\sync_data.ps1 -Upload -Download     # Carica e scarica"
    Write-Host "  .\azure\sync_data.ps1 -Start                # Avvia VM"
    Write-Host "  .\azure\sync_data.ps1 -Stop                 # Spegni VM (stop costi)"
    Write-Host "  .\azure\sync_data.ps1 -UploadDatasetToBlob  # Usa Blob Storage"
}
