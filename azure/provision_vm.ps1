# =============================================================================
# azure/provision_vm.ps1
# =============================================================================
# Crea una VM Azure con GPU NVIDIA T4 pronta per Blender + training PyTorch.
#
# PRE-REQUISITI (eseguire una volta sola in locale):
#   winget install Microsoft.AzureCLI
#   az login
#
# UTILIZZO:
#   cd AI-CHALLENGE
#   .\azure\provision_vm.ps1
#
# Per personalizzare risorse/regione modifica le variabili nella sezione CONFIG.
# =============================================================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# =============================================================================
# CONFIG — modifica qui se vuoi cambiare nomi / region / taglia VM
# =============================================================================

$RESOURCE_GROUP   = "ai-challenge-rg"
$LOCATION         = "eastus"          # Regioni con T4: eastus, westeurope, southeastasia

# VM NC4as_T4_v3  =  4 vCPU, 28 GB RAM, 1x NVIDIA T4 16 GB VRAM
# Costo indicativo: ~0.52 $/ora (paghi solo quando accesa)
# Per solo Blender CPU (più economico): "Standard_D4s_v3" (~0.19 $/ora)
$VM_SIZE          = "Standard_NC4as_T4_v3"

$VM_NAME          = "blender-gpu-vm"
$ADMIN_USER       = "azureuser"
$SSH_KEY_PATH     = "$HOME\.ssh\azure_ai_challenge"   # Chiave SSH generata automaticamente

$STORAGE_ACCOUNT  = "aichallengedata$(Get-Random -Maximum 9999)"   # Nome univoco
$CONTAINER_NAME   = "dataset"

$NSG_NAME         = "$VM_NAME-nsg"
$VNET_NAME        = "$VM_NAME-vnet"
$SUBNET_NAME      = "default"
$PUBLIC_IP_NAME   = "$VM_NAME-ip"
$NIC_NAME         = "$VM_NAME-nic"
$DISK_SIZE_GB     = 128    # OS disk — aumenta se generi 2000+ immagini

# OS: Ubuntu 22.04 LTS (supporta CUDA 12, Blender 4.x, Python 3.10)
$IMAGE            = "Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest"

# =============================================================================
# FUNZIONI HELPER
# =============================================================================

function Write-Step([string]$msg) {
    Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] $msg" -ForegroundColor Cyan
}

function Check-AzCli {
    try { az --version | Out-Null }
    catch {
        Write-Error "Azure CLI non trovato. Installa con: winget install Microsoft.AzureCLI"
        exit 1
    }
    $account = az account show --query "name" -o tsv 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Non autenticato. Esegui: az login"
        exit 1
    }
    Write-Host "  Subscription attiva: $account" -ForegroundColor Green
}

# =============================================================================
# MAIN
# =============================================================================

Write-Step "Verifica Azure CLI e autenticazione"
Check-AzCli

# 1. Resource Group
Write-Step "Creo Resource Group: $RESOURCE_GROUP ($LOCATION)"
az group create `
    --name $RESOURCE_GROUP `
    --location $LOCATION | Out-Null

# 2. Chiave SSH (se non esiste)
Write-Step "Genero chiave SSH: $SSH_KEY_PATH"
if (-not (Test-Path "$SSH_KEY_PATH")) {
    New-Item -ItemType Directory -Force -Path "$HOME\.ssh" | Out-Null
    ssh-keygen -t ed25519 -f $SSH_KEY_PATH -N '""' -C "azure-ai-challenge"
    Write-Host "  Chiave creata: $SSH_KEY_PATH" -ForegroundColor Green
} else {
    Write-Host "  Chiave esistente riutilizzata: $SSH_KEY_PATH" -ForegroundColor Yellow
}
$SSH_PUB_KEY = Get-Content "$SSH_KEY_PATH.pub"

# 3. NSG (apre solo porta 22 SSH)
Write-Step "Creo Network Security Group"
az network nsg create `
    --resource-group $RESOURCE_GROUP `
    --name $NSG_NAME | Out-Null

az network nsg rule create `
    --resource-group $RESOURCE_GROUP `
    --nsg-name $NSG_NAME `
    --name "AllowSSH" `
    --priority 100 `
    --protocol Tcp `
    --destination-port-ranges 22 `
    --access Allow | Out-Null

# 4. VNet + Subnet
Write-Step "Creo rete virtuale"
az network vnet create `
    --resource-group $RESOURCE_GROUP `
    --name $VNET_NAME `
    --subnet-name $SUBNET_NAME | Out-Null

# 5. IP pubblico (Static per non cambiare dopo reboot)
Write-Step "Creo IP pubblico"
az network public-ip create `
    --resource-group $RESOURCE_GROUP `
    --name $PUBLIC_IP_NAME `
    --allocation-method Static `
    --sku Standard | Out-Null

# 6. NIC
Write-Step "Creo Network Interface"
az network nic create `
    --resource-group $RESOURCE_GROUP `
    --name $NIC_NAME `
    --vnet-name $VNET_NAME `
    --subnet $SUBNET_NAME `
    --public-ip-address $PUBLIC_IP_NAME `
    --network-security-group $NSG_NAME | Out-Null

# 7. VM
Write-Step "Creo VM $VM_NAME ($VM_SIZE) — può richiedere 2-3 minuti..."
az vm create `
    --resource-group $RESOURCE_GROUP `
    --name $VM_NAME `
    --nics $NIC_NAME `
    --image $IMAGE `
    --size $VM_SIZE `
    --os-disk-size-gb $DISK_SIZE_GB `
    --admin-username $ADMIN_USER `
    --ssh-key-values $SSH_PUB_KEY `
    --no-wait `
    --output none

Write-Host "  VM in creazione (asincrona)..." -ForegroundColor Yellow
Write-Host "  Attendo completamento provisioning..."
az vm wait --resource-group $RESOURCE_GROUP --name $VM_NAME --created
Write-Host "  VM creata!" -ForegroundColor Green

# 8. NVIDIA GPU driver extension (solo per VM serie NC)
if ($VM_SIZE -like "*NC*") {
    Write-Step "Installo NVIDIA GPU Driver Extension (background, ~5 min)"
    az vm extension set `
        --resource-group $RESOURCE_GROUP `
        --vm-name $VM_NAME `
        --name NvidiaGpuDriverLinux `
        --publisher Microsoft.HpcCompute `
        --version 1.9 `
        --no-wait | Out-Null
    Write-Host "  Driver GPU in installazione (continuerà dopo SSH)" -ForegroundColor Yellow
}

# 9. Storage Account + Container per i dati
Write-Step "Creo Storage Account: $STORAGE_ACCOUNT"
az storage account create `
    --name $STORAGE_ACCOUNT `
    --resource-group $RESOURCE_GROUP `
    --location $LOCATION `
    --sku Standard_LRS `
    --kind StorageV2 | Out-Null

$STORAGE_KEY = az storage account keys list `
    --resource-group $RESOURCE_GROUP `
    --account-name $STORAGE_ACCOUNT `
    --query "[0].value" -o tsv

az storage container create `
    --name $CONTAINER_NAME `
    --account-name $STORAGE_ACCOUNT `
    --account-key $STORAGE_KEY | Out-Null

Write-Host "  Container '$CONTAINER_NAME' creato" -ForegroundColor Green

# 10. Recupera IP pubblico
$PUBLIC_IP = az network public-ip show `
    --resource-group $RESOURCE_GROUP `
    --name $PUBLIC_IP_NAME `
    --query "ipAddress" -o tsv

# 11. Salva info connessione
$CONFIG_FILE = "$PSScriptRoot\vm_config.env"
@"
# Auto-generated by provision_vm.ps1 — $(Get-Date -Format 'yyyy-MM-dd HH:mm')
RESOURCE_GROUP=$RESOURCE_GROUP
VM_NAME=$VM_NAME
VM_PUBLIC_IP=$PUBLIC_IP
ADMIN_USER=$ADMIN_USER
SSH_KEY_PATH=$SSH_KEY_PATH
STORAGE_ACCOUNT=$STORAGE_ACCOUNT
STORAGE_KEY=$STORAGE_KEY
STORAGE_CONTAINER=$CONTAINER_NAME
LOCATION=$LOCATION
"@ | Set-Content $CONFIG_FILE

# =============================================================================
# RIEPILOGO
# =============================================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  VM PRONTA!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  IP pubblico  : $PUBLIC_IP"
Write-Host "  SSH command  : ssh -i $SSH_KEY_PATH $ADMIN_USER@$PUBLIC_IP"
Write-Host "  Config saved : $CONFIG_FILE"
Write-Host ""
Write-Host "  PASSO SUCCESSIVO — configura la VM:"
Write-Host "  1. ssh -i $SSH_KEY_PATH $ADMIN_USER@$PUBLIC_IP"
Write-Host "  2. (sulla VM) bash setup_vm.sh"
Write-Host ""
Write-Host "  Per SPEGNERE la VM quando non la usi (risparmio costo):"
Write-Host "  az vm deallocate --resource-group $RESOURCE_GROUP --name $VM_NAME"
Write-Host ""
Write-Host "  Per RIACCENDERLA:"
Write-Host "  az vm start --resource-group $RESOURCE_GROUP --name $VM_NAME"
Write-Host ""
Write-Host "  Per ELIMINARE TUTTO (quando hai finito):"
Write-Host "  az group delete --name $RESOURCE_GROUP --yes --no-wait"
Write-Host "============================================================" -ForegroundColor Green
