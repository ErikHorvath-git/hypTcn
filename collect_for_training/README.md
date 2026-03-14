# collect_for_training/

This directory holds raw `.bin` frame files captured from KVM guests for
training the hypTcn TCN model on real (non-synthetic) memory pages.

## Directory layout

```
collect_for_training/
├── normal/       ← idle / benign workload frames
├── malware/      ← generic malware (use more specific labels if possible)
├── shellcode/    ← shellcode injection frames
├── rootkit/      ← rootkit (Diamorphine, Reptile, etc.) frames
├── cryptominer/  ← CPU/memory crypto-mining frames
├── ransomware/   ← file-encryption / C2-exfiltration frames
└── scripts/      ← collection and training helper scripts
```

## Frame binary format

Each `.bin` file contains **one 4108-byte frame**:

| Offset | Size | Type        | Description                         |
|--------|------|-------------|-------------------------------------|
| 0      | 8    | uint64 LE   | Physical address the page was read from |
| 8      | 4    | uint32 LE   | Label ID (see table below)          |
| 12     | 4096 | bytes       | Raw 4 KiB page data                 |

Label IDs:

| ID | Label       |
|----|-------------|
| 0  | normal      |
| 1  | malware     |
| 2  | shellcode   |
| 3  | rootkit     |
| 4  | cryptominer |
| 5  | ransomware  |

## Quick start

```sh
# 1. Get System.map from your KVM guest (needed for OS-layer features)
./tools/get_sysmap.sh hyptcn-guest

# 2. Collect 5 minutes of normal behavior
./collect_for_training/scripts/collect_normal.sh hyptcn-guest 300 200

# 3. Set up guest for malware simulation (run once)
./collect_for_training/scripts/prepare_malware_env.sh hyptcn-guest

# 4. Collect all labels interactively (guided wizard)
./collect_for_training/scripts/collect_all_labels.sh hyptcn-guest

# 5. Check dataset statistics
./collect_for_training/scripts/dataset_stats.sh

# 6. Train on collected data
./collect_for_training/scripts/quick_train.sh
```

## Mock mode (no KVM required)

Test the collection pipeline without a real VM:

```sh
./bin/hyptcn --mock --collect --collect-label normal \
             --collect-duration 30 --interval 100
```

## Training on collected data

```sh
# Load .bin files from collect_for_training/ and train
python training/train.py --data-source collect

# Or with custom settings
python training/train.py \
    --data-source collect \
    --collect-dir collect_for_training/ \
    --epochs 50 \
    --batch-size 32
```

## Recommended dataset sizes

| Label       | Minimum frames | Recommended |
|-------------|---------------|-------------|
| normal      | 3 000         | 10 000+     |
| shellcode   | 1 500         | 5 000+      |
| rootkit     | 1 500         | 5 000+      |
| cryptominer | 1 500         | 5 000+      |
| ransomware  | 1 500         | 5 000+      |

At 200 ms intervals, 3 000 frames = 10 minutes of collection.
