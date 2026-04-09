# Guida Azure — AI Glass Cleanliness Detection

Istruzioni per usare la VM **BOLOGNA-AI-AM-MACHINE** già creata su Azure.

---

## La tua VM

| Campo | Valore |
|---|---|
| Nome | BOLOGNA-AI-AM-MACHINE |
| IP pubblico | **72.146.184.240** |
| OS | Ubuntu 24.04 LTS |
| Size | Standard NV8as v4 (8 vCPU, 28 GB RAM) |
| GPU | AMD Radeon Instinct MI25 — **1/4 GPU, 4 GB vRAM** |
| Resource Group | BOLOGNA-AI-AM-MACHINE_group |
| Status | Running |

---

## Flusso di lavoro (senza Azure CLI locale)

Siccome non puoi usare `az` in locale, usi **Azure Cloud Shell** (terminale browser) per tutto.

```
PC locale                   Azure Cloud Shell              VM (72.146.184.240)
──────────                  ─────────────────              ──────────────────
Carica zip ──────────────>  upload_via_cloudshell.sh ───>  AI-CHALLENGE/
                                                           setup_vm.sh
                                                           run_generation.sh
                            upload_via_cloudshell.sh <───  dataset/ (immagini)
Scarica zip <────────────── Download file
```

---

## Passo 1 — Connettiti alla VM

-Par accedere alla macchina:
1)Caricare .pem sul terminale azure
2)Cercare ip:  curl ifconfig.me e configurarlo nel network come source adress ip
3)chmod 400 /home/andrea/BOLOGNA-AI-AM-MACHINE_key.pem
4)ssh -i /home/andrea/BOLOGNA-AI-AM-MACHINE_key.pem azureuseram@20.9.194.236

-SOLO PRIMA VOLTA:
git clone https://github.com/amancio11/AM1.git
bash azure/setup_vm.sh


bash azure/run_generation.sh 0 50


### Via Azure Cloud Shell

**2a.** Apri Cloud Shell su [shell.azure.com](https://shell.azure.com)

**2b.** Carica i file del progetto tramite l'icona Upload (↑) in Cloud Shell:
- Clicca l'icona **Upload/Download** (↑↓) nella toolbar di Cloud Shell
- Seleziona i file del progetto oppure carica uno ZIP

**2b (alternativa). Se il codice è su GitHub:**
```bash
# In Cloud Shell
git clone https://github.com/TUO_USERNAME/AI-CHALLENGE.git ~/AI-CHALLENGE
```

**2c.** Genera la chiave SSH dalla Cloud Shell e aggiungila alla VM:
```bash
# In Cloud Shell
bash ~/AI-CHALLENGE/azure/upload_via_cloudshell.sh --upload
```

La prima volta lo script genera una chiave SSH e ti dice di aggiungerla alla VM via portale:
1. Vai su portal.azure.com → VM → **"Reset password"** (nel menu a sinistra)
2. Scegli **"Reset SSH public key"**
3. Username: `azureuser`
4. Incolla la chiave pubblica mostrata dallo script
5. Clicca **Update**
6. Riesegui `bash ~/AI-CHALLENGE/azure/upload_via_cloudshell.sh --upload`

---

## Passo 3 — Configura la VM (solo prima volta)

```bash
# SSH sulla VM (da terminale locale o Cloud Shell)
ssh azureuser@72.146.184.240

# Sulla VM:
bash ~/AI-CHALLENGE/azure/setup_vm.sh
```

Installa automaticamente (~15 min):
- ROCm 6.x (driver GPU AMD per HIP/OpenCL)
- Blender 4.2 LTS headless (con supporto HIP)
- Python 3.12 virtualenv + requirements.txt
- PyTorch con ROCm

**Al termine potrebbe servire il reboot:**
```bash
sudo reboot
# Riconnettiti dopo ~1 minuto
ssh azureuser@72.146.184.240
```

---

## Passo 4 — Genera le immagini sintetiche

```bash
# Sulla VM
bash ~/AI-CHALLENGE/azure/run_generation.sh 0 500
```

Il processo gira in **tmux** — rimane attivo anche chiudendo SSH.

### Monitorare il progresso

```bash
# Rientra nel processo Blender
tmux attach -t blender_gen
# Esci senza interrompere: Ctrl+B, poi D

# Conta immagini generate
ls ~/AI-CHALLENGE/dataset/images/*.png | wc -l

# Log in tempo reale
tail -f ~/AI-CHALLENGE/logs/generation_*.log

# Spazio usato
du -sh ~/AI-CHALLENGE/dataset/
```

### Tempi stimati

| Scene | CPU-only (probabile con 4 GB vRAM) | Se HIP funziona |
|---|---|---|
| 100 | ~1.5 ore | ~20 min |
| 500 | ~7 ore | ~100 min |

> Con 4 GB di vRAM Blender Cycles spesso non riesce a caricare la scena in GPU e usa automaticamente la CPU. Per forzare il tentativo GPU vedi il troubleshooting.

---

## Passo 5 — Scarica il dataset

### Via Cloud Shell

```bash
# In Cloud Shell
bash ~/AI-CHALLENGE/azure/upload_via_cloudshell.sh --download
```

Scarica i dati dalla VM in Cloud Shell. Poi dal menu Cloud Shell:
1. Clicca **Upload/Download** (↑↓)
2. Scegli **Download**
3. Inserisci il path del file/cartella mostrato dallo script

Per scaricare come ZIP singolo:
```bash
# In Cloud Shell
cd ~ && zip -r dataset_$(date +%Y%m%d).zip dataset_download_*/
# Poi Download → dataset_YYYYMMDD.zip
```

---

## Gestione costi

### Spegni la VM quando non lavori

Dalla VM:
```bash
# Spegne la VM (dealloca — nessun costo CPU/GPU)
sudo shutdown -h now
```

Dal portale Azure:
1. BOLOGNA-AI-AM-MACHINE → **Stop**

**Quando deallocata:** ~0 costo GPU, piccolo costo disco (~0.02 $/ora).

### Riaccendi la VM

portal.azure.com → VM → **Start**

### Costo stimato Standard NV8as v4

| Scenario | Ore stimate (CPU) | Costo |
|---|---|---|
| 500 immagini | ~7 ore | ~$1.05 |
| 2000 immagini | ~28 ore | ~$4.20 |
| Training 50 epoch (batch 4) | ~5 ore | ~$0.75 |

> Costo orario NV8as v4: ~$0.15/ora (Italy North). Spegni la VM tra un run e l'altro.

---

## Struttura file Azure

```
azure/
├── provision_vm.ps1          # Crea nuova VM (non serve, la tua è già pronta)
├── setup_vm.sh               # Configura la VM (Ubuntu 24.04 + AMD GPU)
├── run_generation.sh         # Lancia generazione Blender headless in tmux
├── upload_via_cloudshell.sh  # Upload/download via Azure Cloud Shell
├── sync_data.ps1             # Sync via CLI locale (richiede az in locale)
├── vm_config.env             # *Auto-generato* — NON committare
└── README_AZURE.md           # Questa guida
```

---

## Troubleshooting

### "Permission denied (publickey)" via SSH
```bash
# Aggiungi la chiave via portale: VM → Reset password → Reset SSH public key
# Oppure usa autenticazione con password se l'hai impostata alla creazione
ssh -o PreferredAuthentications=password azureuser@72.146.184.240
```

### Blender non trova la GPU AMD
```bash
# Verifica che ROCm sia caricato
/opt/rocm/bin/rocminfo | grep "gfx"
# Atteso: gfx900 (MI25)

# Se non disponibile, riavvia e riprova
sudo reboot
```

### Blender usa CPU invece di GPU
```bash
# Forza HIP in Blender
export HSA_OVERRIDE_GFX_VERSION=9.0.0
blender --background --python blender/run_generation.py -- \
    --config configs/blender_config.yaml --start 0 --end 1
```

### Upload lento da Cloud Shell
```bash
# Prova con compressione
rsync -avz --compress-level=9 ...
```

---

## Panoramica del setup

```
Macchina locale (Windows)          VM Azure (Ubuntu 22.04 + T4)
─────────────────────────          ─────────────────────────────
provision_vm.ps1     ──────────>   Crea VM + storage
sync_data.ps1 -Upload ─────────>   Carica codice progetto
                                   setup_vm.sh            (installa Blender, CUDA, Python)
                                   run_generation.sh      (genera 500 immagini con Blender)
sync_data.ps1 -Download <──────    Scarica dataset locale
```

---

## Prerequisiti locali

1. **Azure CLI** — installa con:
   ```powershell
   winget install Microsoft.AzureCLI
   az login
   ```

2. **Git for Windows** (include `ssh`, `scp`, `tar`) — già presente se hai Git installato.

3. **rsync** (opzionale ma consigliato per sync veloce):
   ```powershell
   winget install cwRsync.cwRsync
   ```

---

## Passo 1 — Crea la VM Azure

```powershell
cd AI-CHALLENGE
.\azure\provision_vm.ps1
```

Cosa fa:
- Crea un **Resource Group** `ai-challenge-rg`
- Crea una VM **Standard_NC4as_T4_v3** (1× NVIDIA T4, 4 core, 28 GB RAM) in `eastus`
- Genera una coppia di chiavi SSH in `~/.ssh/azure_ai_challenge`
- Installa automaticamente i **driver GPU NVIDIA**
- Crea uno **Storage Account** per il backup del dataset
- Salva IP e credenziali in `azure/vm_config.env`

**Costo stimato:** ~0.52 $/ora mentre la VM è accesa. Usa `-Stop` quando non lavori!

### VM size alternative (modifica `$VM_SIZE` in provision_vm.ps1)

| Size | GPU | CPU | RAM | $/ora | Quando usare |
|---|---|---|---|---|---|
| `Standard_NC4as_T4_v3` | T4 16 GB | 4 | 28 GB | ~0.52 | **Default — rendering + training** |
| `Standard_NC6s_v3` | V100 16 GB | 6 | 112 GB | ~3.06 | Training intensivo |
| `Standard_D4s_v3` | — | 4 | 16 GB | ~0.19 | Solo Blender CPU (lento) |

---

## Passo 2 — Carica il progetto sulla VM

```powershell
.\azure\sync_data.ps1 -Upload
```

Trasferisce tutto il codice (esclude `.venv`, `dataset/`, checkpoint già addestrati). Al termine mostra i prossimi passi.

---

## Passo 3 — Connettiti alla VM

```powershell
# L'IP è salvato in azure/vm_config.env
$ip = (Get-Content azure\vm_config.env | Select-String "VM_PUBLIC_IP").ToString().Split("=")[1]
ssh -i ~/.ssh/azure_ai_challenge azureuser@$ip
```

---

## Passo 4 — Configura la VM (solo prima volta, ~10 min)

```bash
# Dalla VM
bash ~/AI-CHALLENGE/azure/setup_vm.sh
```

Installa automaticamente:
- CUDA 12.1 + driver NVIDIA
- Blender 4.1.1 headless
- Python 3.10 virtualenv
- `requirements.txt` del progetto
- Shortcut `~/run_generation.sh`

Al termine vedrai un riepilogo con i comandi disponibili.

---

## Passo 5 — Genera le immagini sintetiche

```bash
# Dalla VM
bash ~/AI-CHALLENGE/azure/run_generation.sh 0 500
```

Il processo viene avviato in **tmux** — rimane attivo anche se chiudi SSH.

### Monitorare il progresso

```bash
# Entra nella sessione tmux
tmux attach -t blender_gen
# Esci senza interrompere: Ctrl+B, poi D

# Conta immagini generate
ls ~/AI-CHALLENGE/dataset/images/*.png | wc -l

# Vedi log in tempo reale
tail -f ~/AI-CHALLENGE/logs/generation_*.log
```

### Quanto tempo ci vuole?

| Scene | VM T4 (GPU) | VM D4s (CPU) |
|---|---|---|
| 100 | ~15 min | ~1.5 ore |
| 500 | ~75 min | ~7.5 ore |
| 2000 | ~5 ore | ~30 ore |

> Blender Cycles con GPU è ~6× più veloce che con CPU.

---

## Passo 6 — Scarica il dataset

```powershell
# Dalla macchina locale
.\azure\sync_data.ps1 -Download
```

Scarica solo i file nuovi (incrementale). I file arrivano in `dataset/` locale.

---

## Passo 7 — Addestra il modello (opzionale, sulla VM)

Dopo aver generato abbastanza immagini:

```bash
# Dalla VM
source ~/AI-CHALLENGE/.venv/bin/activate
cd ~/AI-CHALLENGE

python -m src.training.train_multitask \
    --config configs/multitask_config.yaml \
    --device cuda
```

Per mandare il training in background:
```bash
tmux new-session -d -s training \
    "cd ~/AI-CHALLENGE && source .venv/bin/activate && \
     python -m src.training.train_multitask --config configs/multitask_config.yaml --device cuda \
     2>&1 | tee logs/training_$(date +%Y%m%d).log"
```

Scarica il checkpoint addestrato:
```powershell
# Dalla macchina locale
scp -i ~/.ssh/azure_ai_challenge azureuser@<IP>:~/AI-CHALLENGE/checkpoints/multitask/best.pth checkpoints/multitask/
```

---

## Gestione costi

### Spegni la VM quando non la usi

```powershell
.\azure\sync_data.ps1 -Stop
```

oppure direttamente:
```powershell
az vm deallocate --resource-group ai-challenge-rg --name blender-gpu-vm
```

**Quando la VM è deallocata:** nessun costo per CPU/GPU. Paghi solo ~0.02 $/ora per il disco OS.

### Riaccendi la VM

```powershell
.\azure\sync_data.ps1 -Start
```

### Eliminare tutto a fine progetto

```powershell
az group delete --name ai-challenge-rg --yes --no-wait
```

Questo elimina VM, disco, rete, IP, storage account — **nessun costo residuo**.

---

## Comandi sync_data.ps1 — riepilogo

| Comando | Cosa fa |
|---|---|
| `.\azure\sync_data.ps1 -Status` | Mostra stato VM, conteggio immagini, ultime righe log |
| `.\azure\sync_data.ps1 -Upload` | Carica codice progetto sulla VM |
| `.\azure\sync_data.ps1 -Download` | Scarica dataset generato in locale |
| `.\azure\sync_data.ps1 -Upload -Download` | Upload + download in sequenza |
| `.\azure\sync_data.ps1 -Start` | Accende la VM |
| `.\azure\sync_data.ps1 -Stop` | Spegne la VM (nessun costo compute) |
| `.\azure\sync_data.ps1 -UploadDatasetToBlob` | Upload via Blob Storage (alternativo) |

---

## Troubleshooting

### SSH: "Permission denied (publickey)"
```powershell
# Verifica che la chiave SSH sia corretta
Get-Content azure\vm_config.env | Select-String "SSH_KEY"
# Usa quella chiave esplicitamente
ssh -i <percorso_chiave> azureuser@<IP>
```

### "nvidia-smi: command not found" dopo setup
```bash
# I driver potrebbero richiedere reboot
sudo reboot
# Riconnettiti dopo ~1 minuto e riprova
nvidia-smi
```

### Blender: "CUDA device not found"
```bash
# Verifica che i driver siano carichi
nvidia-smi
# Se funziona, forza CUDA in Blender
blender --background -noaudio \
    --python-expr "import bpy; bpy.context.preferences.addons['cycles'].preferences.compute_device_type='CUDA'" \
    --python blender/run_generation.py -- --config configs/blender_config.yaml --start 0 --end 1
```

### Spazio disco pieno
```bash
# Controlla spazio
df -h
du -sh ~/AI-CHALLENGE/dataset/

# Riduci samples Cycles nel config per render più piccoli
# In configs/blender_config.yaml: render.samples: 64  (invece di 128)
```

### VM non raggiungibile dopo riavvio
```powershell
# Recupera l'IP aggiornato
az network public-ip show --resource-group ai-challenge-rg --name blender-gpu-vm-ip --query "ipAddress" -o tsv
```

---

## Struttura file Azure

```
azure/
├── provision_vm.ps1      # Crea VM, rete, storage su Azure (Windows locale)
├── setup_vm.sh           # Configura VM Linux (CUDA, Blender, Python)
├── run_generation.sh     # Lancia generazione Blender headless
├── sync_data.ps1         # Upload/download dati, start/stop VM
├── vm_config.env         # *Auto-generato* — IP, credenziali, storage keys
└── README_AZURE.md       # Questa guida
```

> `vm_config.env` contiene chiavi Azure — è già in `.gitignore`. Non committarlo mai.
