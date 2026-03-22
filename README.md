# hypTcn

Hypervisor-aware security toolkit pre live memory introspection KVM guestov. Číta 4 KiB fyzické pamäťové stránky z bežiacej VM cez libvmi a analyzuje ich PyTorch Temporal Convolutional Network (TCN) — detekuje anomálie a klasifikuje aktivitu (shellcode, rootkit, cryptominer, ransomware, normal).

Guest má **nulový footprint** — žiadny agent, žiadne kernel hooky, nič detekovateľné zvnútra VM. TCN namiesto snapshot klasifikátora preto, že sekvencie 16 stránok zachytávajú temporálne vzory (entropia drift, NOP-sled perzistencia, address-scan správanie) ktoré jednorámový klasifikátor nevidí.

---

## Ako to funguje

```
KVM Guest (hyptcn-guest)
        │
        │  libvmi — fyzická pamäť, bez agenta v guest
        ▼
  Go Scanner (bin/hyptcn)
        │  čítanie 4096B stránok + OS-layer info (procesy, moduly, spojenia)
        │  Unix domain socket  [8B addr | 4B modules | 4B connections | 4096B page]
        ▼
  Python Analyzer (python/analyzer.py)
        │  16-frame sliding window → 20 features/frame
        ▼
  TCN model (3 dilated bloky, 2 hlavy)
        ├─ anomaly score  (0–1, alert ak > 0.85)
        └─ activity class (normal / shellcode / rootkit / cryptominer / ransomware)
```

---

## Požiadavky

**Systém (Fedora):**
```bash
sudo dnf install make gcc go python3 libvmi-devel qemu-kvm libvirt
sudo usermod -aG kvm,libvirt $USER
```

**Python venv:**
```bash
make deps   # vytvorí .venv a nainštaluje torch, numpy
```

---

## Od git clone po spustenie

### 1. Klon a build

```bash
git clone <repo-url> hypTcn
cd hypTcn
make deps        # Python .venv
make             # Go binary → bin/hyptcn
```

### 2. VM setup

Potrebuješ KVM guest `hyptcn-guest` s Debian 12 Bookworm:

```bash
# Base image
wget https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-genericcloud-amd64.qcow2 \
     -O ~/vms/debian-12-base.qcow2

# Overlay disk (zmeny sa ukladajú sem, base zostáva čistý)
qemu-img create -f qcow2 -b ~/vms/debian-12-base.qcow2 -F qcow2 ~/vms/hyptcn-guest.qcow2 20G

# Definuj VM
virsh --connect qemu:///session define configs/hyptcn-guest-template.xml

# Nastav root heslo (cez guestfish, VM musí byť vypnutá)
sudo bash tools/mount_vm.sh
```

### 3. Spusti VM a SSH

```bash
virsh --connect qemu:///session start hyptcn-guest

# Po prvom štarte nastav statickú IP vo VM konzole:
virsh --connect qemu:///session console hyptcn-guest
# Vo VM:
#   ip link set enp1s0 up
#   ip addr add 192.168.122.100/24 dev enp1s0
#   ip route add default via 192.168.122.1
#   ssh-keygen -A && systemctl start ssh

# Skopíruj SSH kľúč
ssh-copy-id root@192.168.122.100

# Nastav persistentnú IP (raz)
ssh root@192.168.122.100 'cat > /etc/systemd/network/10-enp1s0.network << EOF
[Match]
Name=enp1s0
[Network]
Address=192.168.122.100/24
Gateway=192.168.122.1
DNS=192.168.122.1
EOF
systemctl enable systemd-networkd'
```

> **Po každom reštarte hosta** obnov NAT:
> ```bash
> sudo bash tools/fix_nat.sh
> ```

### 4. Skopíruj System.map z VM

libvmi potrebuje System.map pre OS-layer introspekciu (procesy, moduly, TCP spojenia):

```bash
./tools/get_sysmap.sh hyptcn-guest
# Uloží do: configs/hyptcn-guest.sysmap
```

Ak get_sysmap.sh zlyhá (placeholder System.map v cloud image):
```bash
# Vo VM nainštaluj debug kernel package
ssh root@192.168.122.100 'apt-get install -y linux-image-$(uname -r | sed "s/-amd64$/-amd64-dbg/")'

# Skopíruj reálny System.map
scp root@192.168.122.100:/usr/lib/debug/boot/System.map-$(ssh root@192.168.122.100 uname -r) \
    configs/hyptcn-guest.sysmap
```

### 5. Nastav libvmi.conf

```bash
sudo tee /etc/libvmi/libvmi.conf << EOF
hyptcn-guest {
    ostype = "Linux";
    sysmap = "$(pwd)/configs/hyptcn-guest.sysmap";
}
EOF
```

### 6. Natrénuj model

```bash
./train.sh
```

Čo robí:
1. Vygeneruje 4000 syntetických frames (2000 normal + 500 každá malware trieda)
2. Natrénuje TCN na 50 epoch s early stopping
3. Nasadí váhy do `python/model/weights/tcn_weights.pt`

Výsledky aktuálneho modelu:
- class_accuracy: **82.9 %**  |  F1: **1.0**  |  ROC-AUC: **1.0**

### 7. Spusti monitoring

```bash
./run.sh
```

Spustí Python analyzer aj Go scanner naraz. Výstup na stdout:

```json
{"timestamp": 1774219200394, "addr": "0x1000000", "score": 0.0, "alert": false, "activity_class": "normal", "status": "ok"}
```

---

## Každodenný workflow

```bash
# Po reštarte hosta
sudo bash tools/fix_nat.sh
virsh --connect qemu:///session start hyptcn-guest

# Spusti monitoring
./run.sh

# Pretrénuj model (po zbere nových dát alebo zmenách)
./train.sh

# Obnov System.map po upgrade kernelu v guest
./tools/get_sysmap.sh hyptcn-guest
```

---

## Štruktúra projektu

```
hypTcn/
│
├── run.sh                      # Spustí celý pipeline (analyzer + scanner)
├── train.sh                    # Generuje dáta + trénuje + nasadí váhy
├── Makefile                    # Build Go binary, Python venv (make / make deps / make clean)
│
├── cmd/hyptcn/
│   └── main.go                 # Cobra CLI vstupný bod Go scannera
│
├── internal/
│   ├── extractor/
│   │   ├── probe.c             # libvmi C wrapper (open/read/close/procesy/moduly/sieť)
│   │   ├── probe.h
│   │   └── extractor.go        # CGO bridge pre Go
│   └── orchestrator/
│       └── engine.go           # Hlavná slučka: čítanie pamäte → UDS → JSON response
│
├── python/
│   ├── analyzer.py             # asyncio UDS server, 16-frame sliding window
│   ├── requirements.txt
│   └── model/
│       ├── tcn.py              # TCNAnomalyDetector (backbone + anomaly head + class head)
│       ├── features.py         # Extrakcia 20 features z 4096B stránky
│       ├── __init__.py
│       └── weights/
│           └── tcn_weights.pt  # Natrénované váhy (nasadené cez train.sh)
│
├── ml/
│   ├── pipeline/
│   │   ├── generate_data.py    # Generuje syntetické processed .npy dáta priamo
│   │   ├── train.py            # Trénovací skript (načíta processed/, uloží váhy)
│   │   ├── preprocess.py       # Raw .bin frames → windowed numpy arrays
│   │   ├── evaluate.py         # Evaluácia natrénovaného modelu
│   │   └── export.py           # Export váh do deployment formátu
│   ├── data/
│   │   ├── raw/                # Surové .bin frames (label/frameN.bin)
│   │   │   ├── normal/         # Sem kopíruj reálne frames z normálnej VM
│   │   │   ├── shellcode/
│   │   │   ├── rootkit/
│   │   │   ├── cryptominer/
│   │   │   └── ransomware/
│   │   ├── processed/          # Auto-generované numpy arrays (vstup pre train.py)
│   │   │   ├── X_train.npy     #   shape (N, 20, 16) float32
│   │   │   ├── X_val.npy
│   │   │   ├── X_test.npy
│   │   │   ├── y_anomaly_*.npy #   shape (N,) float32 — 0=normal, 1=malware
│   │   │   └── y_class_*.npy   #   shape (N,) int64  — 0..4
│   │   └── synthetic/          # Auto-generované syntetické frames (.npy dicts)
│   └── models/
│       ├── tcn_weights.pt      # Záloha natrénovaných váh
│       └── config.json         # Hyperparametre + tréningové štatistiky
│
├── training/
│   └── synthetic.py            # Generátor syntetických feature vektorov (5 tried)
│                               # Definuje realistické distribúcie pre každú triedu
│
├── configs/
│   └── hyptcn-guest.sysmap     # System.map z KVM guest — obnov po upgrade kernelu!
│
├── collect_for_training/       # Manuálne nazbierané .bin frames z reálnej VM
│   ├── normal/                 # Zbieraj počas normálnej VM aktivity
│   ├── shellcode/              # Zbieraj počas simulovaného shellcode útoku
│   ├── rootkit/
│   ├── cryptominer/
│   └── ransomware/
│
├── tools/
│   ├── fix_nat.sh              # Obnoví iptables MASQUERADE pre VM internet po reštarte
│   ├── fix_network.sh          # Opraví orphaned virbr0 bridge (libvirt network reset)
│   ├── get_sysmap.sh           # Skopíruje System.map z VM na host cez SSH/SCP
│   └── mount_vm.sh             # Mountuje VM disk offline (guestfish) — na debug/password reset
│
├── bin/
│   └── hyptcn                  # Skompilovaný Go binary (po make)
│
└── models/                     # Alias — kompatibilita so starším kódom
    └── tcn_weights.pt
```

---

## CLI flags

```
./bin/hyptcn [flags]
```

| Flag | Default | Popis |
|------|---------|-------|
| `--vm` | — | Názov KVM guest (povinný bez `--mock`) |
| `--sysmap` | — | Cesta k System.map — zapne OS-layer mode |
| `--address` | `0x1000000` | Fyzická adresa na čítanie |
| `--interval` | `500` | ms medzi frame-ami |
| `--proc-interval` | `100` | Každých N frame-ov skenuj procesy/moduly/spojenia |
| `--mock` | false | Mock mode bez KVM (testovanie) |
| `--socket` | `/tmp/hyptcn.sock` | UDS socket pre Python analyzer |

Príklady:
```bash
# Mock mode (bez VM, na testovanie)
./bin/hyptcn --mock --interval 100

# Reálna VM, raw mode (bez sysmap)
./bin/hyptcn --vm hyptcn-guest --interval 500

# Plný OS-layer mode
./bin/hyptcn --vm hyptcn-guest --sysmap configs/hyptcn-guest.sysmap --interval 500
```

---

## Wire Protocol

**Go → Python** (4112 bytov/frame):
```
Offset  Veľkosť  Typ           Obsah
0       8B       uint64 LE     fyzická adresa stránky
8       4B       float32 LE    kernel_module_count_norm (moduly / 200)
12      4B       float32 LE    network_conn_count_norm  (TCP spojenia / 100)
16      4096B    raw bytes     surové dáta stránky
```

**Python → Go** (JSON + `\n`):
```json
{"anomaly_score": 0.0, "status": "warming_up"}
{"anomaly_score": 0.02, "activity_class": "normal", "status": "ok"}
{"anomaly_score": 0.91, "activity_class": "shellcode", "status": "ok"}
```

Status `warming_up` — prvých 15 frame-ov (sliding window sa plní).
`alert: true` — score > 0.85.

---

## Model — TCN Architektúra

**Vstup:** tensor `(batch, 20, 16)` — 20 features × 16 frame-ov

**Backbone:**
- 3× `_TCNBlock` s diláciami [1, 2, 4]
- Každý blok: 2× weight-normalized kauzálny Conv1d (kernel=3, filters=32, dropout=0.1) + residual
- GlobalAvgPool1d → vektor 32

**Výstupné hlavy:**
- Anomaly: `Linear(32→16, ReLU) → Linear(16→1, Sigmoid)` → score ∈ [0,1]
- Class:   `Linear(32→64, ReLU) → Linear(64→5)` → argmax → trieda 0–4

**Triedy:** `0=normal, 1=shellcode, 2=rootkit, 3=cryptominer, 4=ransomware`

**20 features:**

| # | Feature | Popis |
|---|---------|-------|
| 0 | byte_entropy | Shannon entropia (0–8) |
| 1 | nonzero_ratio | Podiel nenulových bajtov |
| 2 | printable_ratio | Podiel ASCII printable znakov |
| 3 | high_byte_ratio | Podiel bajtov > 0x7F |
| 4 | unique_bytes | Počet unikátnych bajtov / 256 |
| 5 | top4_freq | Frekvencia 4 najčastejších bajtov |
| 6 | zero_runs | Pomér nulových sérií |
| 7 | addr_norm | Normalizovaná fyzická adresa |
| 8 | entropy_blocks_std | Std entropie naprieč 16 blokmi |
| 9 | compression_ratio | Odhadovaný kompresný pomer |
| 10 | null_run_ratio | Podiel nulových behov |
| 11 | pe_header_score | Skóre PE hlavičky (Windows exec) |
| 12 | elf_header_score | Skóre ELF hlavičky (Linux exec) |
| 13 | syscall_pattern_count | Počet syscall vzorov / norm |
| 14 | nop_sled_score | Detekcia NOP sled (0x90 sekvencie) |
| 15 | string_density | Hustota ASCII stringov |
| 16 | entropy_delta | Zmena entropie vs. predch. frame |
| 17 | addr_delta | Zmena adresy vs. predch. frame |
| 18 | kernel_module_count_norm | Počet kernel modulov / 200 |
| 19 | network_conn_count_norm | Počet TCP spojení / 100 |

---

## VM — Technické detaily

| Parameter | Hodnota |
|-----------|---------|
| Guest name | `hyptcn-guest` |
| Disk | `~/vms/hyptcn-guest.qcow2` (overlay na `debian-12-base.qcow2`) |
| Kernel | `6.1.0-42-cloud-amd64` (Debian 12 Bookworm) |
| Root heslo | `hyptcn` |
| IP | `192.168.122.100` (statická, `/etc/systemd/network/10-enp1s0.network`) |
| Bridge | `virbr0` — `192.168.122.1/24` |
| libvirt | `qemu:///session` (user session, nie system) |
| System.map | `configs/hyptcn-guest.sysmap` — 3.5 MB reálny (z `/usr/lib/debug/boot/`) |

> Libvirt `default` network je `inactive` — `virbr0` existuje cez NetworkManager ale DHCP nefunguje. VM má statickú IP a NAT treba obnoviť po reštarte hosta: `sudo bash tools/fix_nat.sh`

---

## Zber reálnych tréningových dát

```bash
# Počas normálnej aktivity
./bin/hyptcn --vm hyptcn-guest --sysmap configs/hyptcn-guest.sysmap \
    --interval 100 --log-dir collect_for_training/normal

# Počas simulovaného útoku (iný terminál)
./bin/hyptcn --vm hyptcn-guest --log-dir collect_for_training/shellcode

# Po nazbieraní dát, pretrénuj
./train.sh
```

Frames sa ukladajú ako `NNNNN_AAAAAAAAAAAAAAAA.bin` (frame index + hex adresa stránky).

---

## Zhrnutie príkazov

```bash
make deps                                    # Python venv + torch
make                                         # Build Go binary
./train.sh                                   # Generuj dáta + trénuj + nasaď váhy
./run.sh                                     # Spusti monitoring pipeline
sudo bash tools/fix_nat.sh                   # Obnov VM internet po reštarte hosta
./tools/get_sysmap.sh hyptcn-guest           # Obnov System.map z VM
ssh root@192.168.122.100                     # SSH do VM
virsh --connect qemu:///session list --all   # Zoznam VM
make clean                                   # Vyčisti build artefakty
```
