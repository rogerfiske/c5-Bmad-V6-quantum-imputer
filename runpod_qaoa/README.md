# QAOA for QS/QV Prediction
## Dr. Jack Hammer - QMLE Director

**Quantum Approximate Optimization Algorithm** implementation for predicting least-likely QV column activations in the c5_Matrix dataset.

---

## Overview

This package implements QAOA with an **XY-mixer** that naturally preserves Hamming weight, matching the dataset's constraint of exactly 5 activations per event. The algorithm optimizes a cost Hamiltonian encoding:
- **Inverse frequency**: Prefer columns that are rarely activated
- **Anti-co-occurrence**: Prefer column combinations that don't usually appear together

### Key Features
- XY-mixer for Hamming weight preservation
- PennyLane with lightning.gpu backend (cuQuantum)
- COBYLA/SPSA optimizers for noise-robust optimization
- SHA-512 provenance tracking for reproducibility

---

## Hardware Requirements

### RunPod Configuration
- **Template**: RunPod Pytorch 2.8.0
- **GPU**: 2x NVIDIA H200 (141GB each)
- **vCPU**: 24+
- **RAM**: 250GB+
- **Storage**: 50GB+

### Why 2x H200?
- 39 qubits = 2^39 complex amplitudes = ~4TB state vector
- Distributed across 2 GPUs via cuQuantum
- Single H200 insufficient for full state vector

---

## Quick Start

### 1. Deploy on RunPod

1. Create a new Pod with **RunPod Pytorch 2.8.0** template
2. Select **2x NVIDIA H200** GPUs
3. Upload the `runpod_qaoa.zip` package to `/workspace/`
4. Unzip:
   ```bash
   cd /workspace
   unzip runpod_qaoa.zip
   cd runpod_qaoa
   ```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: cuQuantum is typically pre-installed on RunPod GPU instances. If not:
```bash
pip install cuquantum-python pennylane-lightning-gpu
```

### 3. Run QAOA

**Option A: Command Line**
```bash
python run_qaoa.py --data data/c5_Matrix.csv --layers 8 --shots 8192 --iterations 500
```

**Option B: Jupyter Notebook**
```bash
jupyter notebook run_qaoa_notebook.ipynb
```

---

## Directory Structure

```
runpod_qaoa/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── run_qaoa.py               # Main execution script
├── run_qaoa_notebook.ipynb   # Interactive Jupyter notebook
├── data/
│   └── c5_Matrix.csv         # Dataset (11,622 events x 39 columns)
├── src/
│   ├── __init__.py           # Module exports
│   ├── qaoa_config.py        # Configuration dataclasses
│   ├── data_loader.py        # Dataset loading and Hamiltonian weights
│   ├── qaoa_circuit.py       # QAOA circuit with XY-mixer
│   ├── optimizer.py          # COBYLA/SPSA optimizers
│   └── utils.py              # Predictions and provenance utilities
└── results/                  # Output directory (created on run)
    ├── qaoa_provenance_*.json
    ├── qaoa_optimization_history_*.json
    └── qaoa_analysis_*.json
```

---

## Configuration Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `data/c5_Matrix.csv` | Path to dataset |
| `--layers` | `8` | QAOA layers (p parameter) |
| `--shots` | `8192` | Measurement shots |
| `--iterations` | `500` | Max optimization iterations |
| `--optimizer` | `COBYLA` | Optimizer: COBYLA, SPSA, BFGS |
| `--device` | `lightning.gpu` | PennyLane device |
| `--output` | `results` | Output directory |
| `--seed` | `42` | Random seed |

### Configuration Dataclass

Edit `src/qaoa_config.py` for advanced configuration:

```python
QAOAConfig(
    n_qubits=39,          # Number of qubits (= columns)
    n_layers=8,           # QAOA depth
    mixer_type='xy',      # 'xy' or 'x' mixer
    n_select=20,          # Columns to predict
    w_freq=1.0,           # Frequency weight in Hamiltonian
    w_cooc=0.5,           # Co-occurrence penalty weight
    optimizer='COBYLA',   # Optimization algorithm
    max_iterations=500,   # Max iterations
    n_shots=8192,         # Measurement shots
    seed=42               # Random seed
)
```

---

## QAOA Theory

### Cost Hamiltonian

$$H_C = -\sum_i h_i Z_i - \sum_{i<j} J_{ij} Z_i Z_j$$

Where:
- $h_i = 1 - \text{freq}_i$: Higher for less frequent columns
- $J_{ij} = -\text{cooc}_{ij}$: Negative for commonly co-occurring pairs

### XY-Mixer

$$H_B = \sum_{i<j} (X_i X_j + Y_i Y_j)$$

The XY-mixer preserves Hamming weight (number of |1⟩ states), naturally enforcing the constraint that we select a fixed number of columns.

### QAOA Ansatz

$$|\psi(\gamma, \beta)\rangle = U_B(\beta_p) U_C(\gamma_p) \cdots U_B(\beta_1) U_C(\gamma_1) |+\rangle^n$$

---

## Expected Output

### Console Output
```
QAOA FOR QS/QV PREDICTION
Dr. Jack Hammer - QMLE Director
======================================================================

STEP 1: Loading and preparing data
Dataset shape: 11622 events x 39 columns
Constraint verified: Exactly 5 activations per event
...

STEP 4: Running QAOA optimization
Starting COBYLA optimization...
  Initial cost: -0.234567
  Iteration 10: cost = -0.345678 (best = -0.345678)
  ...

QAOA PREDICTION RESULTS
======================================================================
Top 20 Least Likely QV Columns:
  1. QV_23  (QAOA selection frequency: 0.8234)
  2. QV_17  (QAOA selection frequency: 0.7891)
  ...
```

### Output Files

1. **qaoa_provenance_*.json**: Full experiment record with SHA-512 hashes
2. **qaoa_optimization_history_*.json**: Parameter evolution during optimization
3. **qaoa_analysis_*.json**: Sample statistics and column frequencies

---

## Troubleshooting

### "Failed to create lightning.gpu"
- Ensure cuQuantum is installed: `pip install cuquantum-python pennylane-lightning-gpu`
- Check GPU availability: `nvidia-smi`
- Falls back to `default.qubit` (CPU) automatically

### Out of Memory
- Reduce `n_shots` (e.g., 4096 instead of 8192)
- Use single GPU mode (slower but less memory)
- Consider layer-by-layer simulation

### Slow Optimization
- SPSA is faster but noisier than COBYLA
- Reduce `max_iterations` for quick tests
- Use fewer layers (p=4 instead of p=8)

---

## Provenance and Reproducibility

Every run generates a full provenance record with:
- SHA-512 hash of input dataset
- SHA-512 hash of all results
- Complete configuration snapshot
- Timestamps and researcher attribution

Example provenance entry:
```json
{
  "experiment_id": "QAOA-QMLE-2024-12-02",
  "researcher": "Dr. Jack Hammer (QMLE Director)",
  "dataset": {
    "hash_sha512": "a1b2c3..."
  },
  "result_hash_sha512": "d4e5f6..."
}
```

---

## Contact

**Dr. Jack Hammer**
QMLE Director
Quantum Machine Learning Engineering Division

---

*Generated by QMLE Framework v1.0.0*
