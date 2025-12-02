"""
Data Loader for QS/QV Dataset
Dr. Jack Hammer - QMLE Director

Load and preprocess the c5_Matrix.csv dataset for QAOA.
Compute column frequencies and co-occurrence matrix for Hamiltonian.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from hashlib import sha512


def load_dataset(filepath: str) -> Tuple[np.ndarray, Dict]:
    """
    Load the QS/QV dataset and compute statistics.

    Args:
        filepath: Path to c5_Matrix.csv

    Returns:
        events: Binary matrix (n_events, n_columns)
        stats: Dictionary with column frequencies, co-occurrence, etc.
    """
    print(f"Loading dataset: {filepath}")

    # Load CSV
    df = pd.read_csv(filepath)

    # Extract binary matrix (skip event-ID column)
    events = df.iloc[:, 1:].values.astype(np.float32)
    n_events, n_columns = events.shape

    print(f"Dataset shape: {n_events} events x {n_columns} columns")

    # Verify constraint: exactly 5 activations per event
    activations_per_event = events.sum(axis=1)
    assert np.allclose(activations_per_event, 5.0), \
        f"Expected exactly 5 activations per event, got mean={activations_per_event.mean()}"
    print(f"Constraint verified: Exactly 5 activations per event")

    # Compute column frequencies
    column_frequencies = events.mean(axis=0)
    print(f"Column frequency range: {column_frequencies.min():.4f} - {column_frequencies.max():.4f}")

    # Compute co-occurrence matrix (normalized)
    # cooc[i,j] = P(column i AND column j active)
    cooccurrence = np.dot(events.T, events) / n_events
    np.fill_diagonal(cooccurrence, 0)  # Ignore self-activation

    # Compute expected co-occurrence under independence
    expected_cooc = np.outer(column_frequencies, column_frequencies)
    np.fill_diagonal(expected_cooc, 0)

    # Deviation from independence
    cooc_deviation = cooccurrence - expected_cooc

    # Compute dataset hash for provenance
    with open(filepath, 'rb') as f:
        dataset_hash = sha512(f.read()).hexdigest()

    stats = {
        'n_events': n_events,
        'n_columns': n_columns,
        'column_frequencies': column_frequencies,
        'cooccurrence': cooccurrence,
        'expected_cooccurrence': expected_cooc,
        'cooc_deviation': cooc_deviation,
        'sparsity': 1 - column_frequencies.mean(),
        'dataset_hash': dataset_hash
    }

    print(f"Sparsity: {stats['sparsity']:.2%}")
    print(f"Dataset hash: {dataset_hash[:32]}...")

    return events, stats


def get_column_rankings(stats: Dict, n_select: int = 20) -> np.ndarray:
    """
    Get baseline ranking of columns by frequency (least frequent first).

    This provides a warm-start initialization for QAOA.

    Args:
        stats: Dataset statistics from load_dataset()
        n_select: Number of columns to select

    Returns:
        indices: Array of column indices (0-indexed), sorted by frequency ascending
    """
    column_frequencies = stats['column_frequencies']

    # Sort ascending (least frequent first)
    ranking = np.argsort(column_frequencies)

    # Return top n_select least frequent
    return ranking[:n_select]


def compute_hamiltonian_weights(
    stats: Dict,
    w_freq: float = 1.0,
    w_cooc: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute weights for QAOA cost Hamiltonian.

    Cost Hamiltonian: H_C = -sum_i h_i * Z_i - sum_{i<j} J_ij * Z_i * Z_j

    We want to MAXIMIZE H_C for least-likely columns, so:
    - h_i = weight for selecting column i (higher = more desirable)
    - J_ij = interaction between columns i and j

    For "least likely" prediction:
    - h_i should be higher for LESS frequent columns
    - J_ij should penalize pairs that often co-occur (we want rare combinations)

    Args:
        stats: Dataset statistics
        w_freq: Weight for frequency term
        w_cooc: Weight for co-occurrence term

    Returns:
        h: Single-qubit weights (n_columns,)
        J: Two-qubit couplings (n_columns, n_columns)
    """
    column_frequencies = stats['column_frequencies']
    cooccurrence = stats['cooccurrence']
    n_columns = stats['n_columns']

    # Single-qubit terms: h_i = 1 - freq_i (inverse frequency)
    # Higher weight for less frequent columns
    h = w_freq * (1.0 - column_frequencies)

    # Normalize to [-1, 1] range for numerical stability
    h = h / h.max()

    # Two-qubit terms: J_ij = -cooc_ij (negative = penalize co-occurrence)
    # We want columns that DON'T usually appear together
    J = -w_cooc * cooccurrence

    # Normalize
    J_max = np.abs(J).max()
    if J_max > 0:
        J = J / J_max

    print(f"Hamiltonian weights computed:")
    print(f"  h range: [{h.min():.4f}, {h.max():.4f}]")
    print(f"  J range: [{J.min():.4f}, {J.max():.4f}]")

    return h, J


def prepare_qaoa_data(config) -> Dict:
    """
    Prepare all data needed for QAOA execution.

    Args:
        config: QAOAConfig instance

    Returns:
        Dictionary with all QAOA inputs
    """
    events, stats = load_dataset(config.data_path)

    h, J = compute_hamiltonian_weights(
        stats,
        w_freq=config.w_freq,
        w_cooc=config.w_cooc
    )

    baseline_ranking = get_column_rankings(stats, config.n_select)

    print(f"\nBaseline 1 ranking (top {config.n_select} least frequent):")
    column_names = [f"QV_{i+1}" for i in baseline_ranking]
    print(f"  {', '.join(column_names)}")

    return {
        'events': events,
        'stats': stats,
        'h': h,
        'J': J,
        'baseline_ranking': baseline_ranking,
        'n_qubits': config.n_qubits,
        'n_select': config.n_select
    }
