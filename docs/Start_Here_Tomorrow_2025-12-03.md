# Start Here Tomorrow - December 3, 2025
## Dr. Jack Hammer - QMLE Director
## QAOA Tensor Network Implementation

---

## Quick Context
Yesterday we built a complete QAOA implementation for QS/QV prediction, but discovered that **39 qubits requires 8 TB of GPU RAM** - impossible even with 2x H200 GPUs. Today we implement tensor network simulation which can handle all 39 qubits.

---

## Today's Goals

### Goal 1: Tensor Network QAOA (PRIMARY)
Implement QAOA using Matrix Product State (MPS) tensor network representation.

### Goal 2: Shot-Based Simulation (COMPARISON)
Implement trajectory-based sampling for comparison.

### Goal 3: Compare Results
Run both methods and compare predictions against Baseline 1.

---

## Step-by-Step Instructions

### Step 1: Create Tensor Network Version

Ask Claude to create `run_qaoa_tensor.py` with these specifications:

```
Create a tensor network version of QAOA for all 39 qubits using:
- PennyLane's default.tensor device with MPS method
- Bond dimension parameter (start with 64, can increase)
- Same cost Hamiltonian and XY-mixer as original
- All 39 QV columns (no reduction needed)
```

**Key PennyLane code pattern:**
```python
import pennylane as qml

# Tensor network device - handles large qubit counts
dev = qml.device("default.tensor", wires=39, method="mps", max_bond_dim=64)
```

### Step 2: Update Repository

After creating the new file:
```bash
cd C:\Users\Minis\CascadeProjects\c5-Bmad-V6-quantum-imputer
git add runpod_qaoa/run_qaoa_tensor.py
git commit -m "Add tensor network QAOA for 39 qubits"
git push
```

### Step 3: Deploy to RunPod

On RunPod terminal:
```bash
cd /workspace/c5-Bmad-V6-quantum-imputer
git pull

# Install tensor network backend if needed
pip install quimb cotengra

cd runpod_qaoa
python run_qaoa_tensor.py --qubits 39 --layers 6 --bond-dim 64
```

### Step 4: Run Comparison (Optional)

If time permits, create shot-based version and compare:
- Tensor network predictions
- Shot-based predictions
- Baseline 1 frequency ranking
- Overlap analysis

---

## Expected Outputs

### From Tensor Network Run:
```
results/
├── qaoa_tensor_provenance_*.json    # Full experiment record
├── qaoa_tensor_predictions_*.json   # Top 20 predicted columns
└── qaoa_tensor_analysis_*.json      # Optimization history
```

### Prediction Format:
```
Top 20 Least Likely QV Columns (Tensor Network):
  1. QV_XX (frequency: 0.XXXX)
  2. QV_XX (frequency: 0.XXXX)
  ...
```

---

## Technical Reference

### Tensor Network Parameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `max_bond_dim` | 64-256 | Higher = more accurate, more memory |
| `cutoff` | 1e-10 | Singular value cutoff for truncation |
| `method` | "mps" | Matrix Product State representation |

### Memory Estimates (MPS)

| Bond Dim | Approximate Memory | Accuracy |
|----------|-------------------|----------|
| 32       | ~2 GB             | Low      |
| 64       | ~8 GB             | Medium   |
| 128      | ~32 GB            | High     |
| 256      | ~128 GB           | Very High|

### Why Tensor Networks Work for QAOA
- QAOA circuits have **limited entanglement** (shallow depth)
- MPS efficiently represents low-entanglement states
- Bond dimension controls accuracy vs memory trade-off
- For 6-8 QAOA layers, bond dim 64-128 should suffice

---

## Troubleshooting

### If `default.tensor` not available:
```bash
pip install pennylane[tensor]
# or
pip install quimb cotengra
```

### If bond dimension too low (poor results):
- Increase `max_bond_dim` to 128 or 256
- Check entanglement entropy in output
- May need more QAOA layers for convergence

### If still hitting memory limits:
- Reduce bond dimension
- Use `cutoff` parameter more aggressively
- Consider hybrid approach (tensor network + sampling)

---

## Files to Reference

| File | Purpose |
|------|---------|
| `runpod_qaoa/src/qaoa_circuit.py` | XY-mixer and cost layer implementations |
| `runpod_qaoa/src/data_loader.py` | Hamiltonian weight computation |
| `runpod_qaoa/src/optimizer.py` | COBYLA/SPSA optimization |
| `docs/Session_Summary_2025-12-02.md` | Yesterday's detailed summary |

---

## Success Criteria

1. **Tensor network runs on all 39 qubits** without memory error
2. **Optimization converges** (cost decreases over iterations)
3. **Predictions generated** for top 20 least-likely columns
4. **Comparison with Baseline 1** shows meaningful difference or agreement

---

## Questions to Answer Today

1. Does tensor network QAOA produce different predictions than Baseline 1?
2. What bond dimension is sufficient for this problem?
3. How does computation time compare to what state-vector would have been?
4. Are the XY-mixer dynamics captured accurately by MPS?

---

*Ready to start: December 3, 2025*
*Dr. Jack Hammer - QMLE Director*

---

## Quick Start Command

When ready, tell Claude:

> "Create a tensor network version of QAOA (run_qaoa_tensor.py) using PennyLane's default.tensor device with MPS method. It should handle all 39 qubits with configurable bond dimension. Use the same cost Hamiltonian and XY-mixer from the existing implementation."
