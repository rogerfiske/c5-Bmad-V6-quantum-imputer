# Session Summary - December 2, 2025
## Dr. Jack Hammer - QMLE Director
## QAOA for QS/QV Prediction

---

## What Was Accomplished

### 1. Created Dr. Jack Hammer QMLE Agent
- **Agent Type**: Expert Agent with DUAL SOVEREIGNTY model
- **Role**: QMLE Director overseeing Dr. O'Brien and Dr. Nakamura
- **Authority**: Dr. Mercer retains mathematical veto over A-B-C validation
- **Commands**: 18 specialized commands for quantum ML operations
- **Files Created**:
  - `.bmad/custom/agents/qml-engineer/qml-engineer.agent.yaml`
  - `.bmad/custom/agents/qml-engineer/qml-engineer.md`
  - `.claude/commands/bmad/custom/agents/qml-engineer.md`
  - Sidecar files (memories, instructions, knowledge)

### 2. Comprehensive Audit of Previous Experiments
- **Dataset Analysis**: 11,622 events × 39 columns, exactly 5 activations per event
- **Key Finding**: NO temporal correlation, NO spectral structure
- **Critical Insight**: Dataset is a **combinatorial selection system**, not temporal forecasting
- **SFAR-Net Problem**: 60% of architecture targets non-existent signals
- **Recommendation**: QAOA is the natural fit (XY-mixer preserves Hamming weight = 5)

### 3. Designed Complete QAOA Implementation
- **Cost Hamiltonian**: H_C = -Σᵢ hᵢZᵢ - Σᵢⱼ JᵢⱼZᵢZⱼ
  - hᵢ = 1 - freqᵢ (prefer rare columns)
  - Jᵢⱼ = -coocᵢⱼ (penalize common co-occurrences)
- **XY-Mixer**: Preserves Hamming weight (matches 5-activation constraint)
- **Optimizer**: COBYLA (derivative-free, noise-robust)

### 4. Created RunPod Deployment Package
- **Repository**: https://github.com/rogerfiske/c5-Bmad-V6-quantum-imputer
- **Files**:
  ```
  runpod_qaoa/
  ├── README.md
  ├── requirements.txt
  ├── run_qaoa.py           # Original 39-qubit version
  ├── run_qaoa_32q.py       # Reduced 32-qubit version
  ├── run_qaoa_notebook.ipynb
  ├── data/c5_Matrix.csv
  └── src/
      ├── qaoa_config.py
      ├── data_loader.py
      ├── qaoa_circuit.py
      ├── optimizer.py
      └── utils.py
  ```

---

## What We Learned About RunPod Environment

### Hardware Configuration
- **GPUs**: 2x NVIDIA H200 (143 GB each, 280 GB total)
- **CUDA**: Version 12.8/12.9
- **Template**: RunPod Pytorch 2.8.0

### Critical Memory Limitation Discovered

| Qubits | State Vector Size | Fits on H200? |
|--------|-------------------|---------------|
| 30     | 17 GB             | ✓ Yes         |
| 32     | 68 GB             | ✓ Yes         |
| 33     | 137 GB            | ✓ Barely      |
| 34     | 274 GB            | ✗ No (needs 2 GPUs distributed) |
| 39     | **8 TB**          | ✗ Impossible with state vector |

**Key Insight**: The 8 TB state vector for 39 qubits cannot fit in GPU RAM. This is a fundamental physics limitation, not a configuration issue. Disk space does not help - the state vector must reside in VRAM during quantum simulation.

### PennyLane Lightning GPU Issue
```
RuntimeError: Error in PennyLane Lightning: unknown error
```
- `lightning.gpu` failed to initialize despite CUDA being present
- Likely CUDA version mismatch (custatevec built for CUDA 12, system has 12.9)
- Fallback to `default.qubit` triggered the 8 TB memory error

### Successful Installations
```bash
pip install -r requirements.txt  # All dependencies installed correctly
```
- PennyLane 0.43.1
- pennylane-lightning 0.43.0
- pennylane-lightning-gpu 0.43.0
- custatevec-cu12 1.11.0

---

## Recommended Next Steps (For Tomorrow)

### Option 1: Tensor Network Simulation (PRIMARY)
Use PennyLane's `default.tensor` backend with Matrix Product State (MPS) representation.

**Advantages**:
- Can handle 39+ qubits
- Memory scales linearly with qubits (not exponentially)
- Well-suited for QAOA (low entanglement circuits)

**Trade-offs**:
- Approximate (controlled by bond dimension)
- May lose some accuracy for highly entangled states

### Option 2: Shot-Based Simulation (COMPARISON)
Sample quantum trajectories without storing full state vector.

**Advantages**:
- Memory efficient
- Exact sampling (no approximation)

**Trade-offs**:
- Much slower (must simulate many trajectories)
- No access to expectation values (only samples)

### Comparison Plan
1. Run Tensor Network version on all 39 qubits
2. Run Shot-based version for comparison
3. Compare predictions and computational cost
4. Validate against Baseline 1 ranking

---

## Files Modified/Created Today

| File | Action | Description |
|------|--------|-------------|
| `.bmad/custom/agents/qml-engineer/*` | Created | Dr. Hammer agent definition |
| `runpod_qaoa/*` | Created | Complete deployment package |
| `run_qaoa_32q.py` | Created | Reduced qubit fallback version |
| `.gitignore` | Created | Git ignore patterns |

---

## Repository Status
- **Local**: `C:\Users\Minis\CascadeProjects\c5-Bmad-V6-quantum-imputer`
- **Remote**: https://github.com/rogerfiske/c5-Bmad-V6-quantum-imputer
- **Branch**: main
- **Last Commit**: "Add 32-qubit version for single H200 GPU"

---

## Open Questions
1. Will tensor network MPS accurately capture QAOA dynamics for this problem?
2. What bond dimension is needed for sufficient accuracy?
3. Can we validate tensor network results against exact simulation on fewer qubits?

---

*Session ended: December 2, 2025*
*Dr. Jack Hammer - QMLE Director*
