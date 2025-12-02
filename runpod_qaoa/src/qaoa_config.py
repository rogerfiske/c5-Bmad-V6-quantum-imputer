"""
QAOA Configuration for QS/QV Prediction
Dr. Jack Hammer - QMLE Director

Configuration constants and hyperparameters for QAOA circuit.
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class QAOAConfig:
    """QAOA hyperparameters and settings."""

    # Dataset
    data_path: str = "data/c5_Matrix.csv"
    n_columns: int = 39  # Number of QV columns
    n_select: int = 20   # Number of columns to predict as "least likely"

    # QAOA Circuit Parameters
    n_qubits: int = 39   # One qubit per column
    n_layers: int = 8    # QAOA depth (p) - start with 8, can increase

    # Mixer Type
    # 'xy' = XY-mixer (preserves Hamming weight)
    # 'x' = Standard X-mixer (allows any Hamming weight)
    mixer_type: str = 'xy'

    # Hamiltonian Weights
    # w_freq: Weight for column frequency term (higher = prioritize rare columns)
    # w_cooc: Weight for co-occurrence term (higher = penalize co-occurring pairs)
    w_freq: float = 1.0
    w_cooc: float = 0.5

    # Optimization
    optimizer: str = 'COBYLA'  # COBYLA, SPSA, or Adam
    max_iterations: int = 500
    convergence_tol: float = 1e-6

    # Simulation
    n_shots: int = 8192  # Measurement shots per circuit evaluation
    use_gpu: bool = True  # Use cuQuantum GPU acceleration

    # Random seed for reproducibility
    seed: int = 42

    # Output
    results_dir: str = "results"

    # Hardware
    n_gpus: int = 2  # Number of H200 GPUs (2x recommended for 39 qubits)


@dataclass
class PennyLaneConfig:
    """PennyLane-specific configuration for cuQuantum."""

    # Device selection
    # 'lightning.gpu' for cuQuantum GPU simulation
    # 'default.qubit' for CPU fallback
    device: str = 'lightning.gpu'

    # cuQuantum settings
    mpi_enabled: bool = False  # Enable for multi-GPU with MPI

    # Batch size for parameter-shift gradients
    batch_size: int = 32

    # Memory optimization
    adjoint_diff: bool = True  # Use adjoint differentiation (memory efficient)


# Global configuration instances
QAOA_CONFIG = QAOAConfig()
PENNYLANE_CONFIG = PennyLaneConfig()


def get_device_string(config: PennyLaneConfig, n_qubits: int) -> str:
    """Get PennyLane device string based on configuration."""
    if config.device == 'lightning.gpu':
        return f"lightning.gpu"
    else:
        return "default.qubit"


def print_config():
    """Print current configuration for logging."""
    print("=" * 70)
    print("QAOA CONFIGURATION")
    print("=" * 70)
    print(f"Dataset: {QAOA_CONFIG.data_path}")
    print(f"Qubits: {QAOA_CONFIG.n_qubits}")
    print(f"QAOA Layers (p): {QAOA_CONFIG.n_layers}")
    print(f"Mixer Type: {QAOA_CONFIG.mixer_type}")
    print(f"Optimizer: {QAOA_CONFIG.optimizer}")
    print(f"Max Iterations: {QAOA_CONFIG.max_iterations}")
    print(f"Shots: {QAOA_CONFIG.n_shots}")
    print(f"GPU Enabled: {QAOA_CONFIG.use_gpu}")
    print(f"Seed: {QAOA_CONFIG.seed}")
    print("=" * 70)
