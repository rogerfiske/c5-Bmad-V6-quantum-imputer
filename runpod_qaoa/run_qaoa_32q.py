#!/usr/bin/env python3
"""
QAOA Main Execution Script - 32 Qubit Version
Dr. Jack Hammer - QMLE Director

Reduced qubit version that fits on single H200 GPU (143GB).
32 qubits = 68GB state vector.

Selects 32 most informative columns based on frequency variance.
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.qaoa_config import QAOAConfig
from src.data_loader import load_dataset, compute_hamiltonian_weights
from src.qaoa_circuit import (
    create_qaoa_qnode,
    create_sampling_qnode,
    get_initial_params
)
from src.optimizer import run_optimization
from src.utils import (
    samples_to_predictions,
    evaluate_prediction,
    generate_provenance,
    save_results,
    print_prediction_summary
)


def select_top_columns(stats, n_select=32):
    """
    Select top N columns based on frequency deviation from mean.

    We want columns that are either very rare or very common,
    as these are most informative for prediction.
    """
    freqs = stats['column_frequencies']
    mean_freq = freqs.mean()

    # Score by absolute deviation from mean (most distinctive columns)
    deviation = np.abs(freqs - mean_freq)

    # Get indices sorted by deviation (highest first)
    selected_indices = np.argsort(-deviation)[:n_select]

    # Sort selected indices for consistent ordering
    selected_indices = np.sort(selected_indices)

    return selected_indices


def main():
    """Main QAOA execution pipeline - 32 qubit version."""

    print("=" * 70)
    print("QAOA FOR QS/QV PREDICTION - 32 QUBIT VERSION")
    print("Dr. Jack Hammer - QMLE Director")
    print("=" * 70)
    print(f"Execution started: {datetime.now().isoformat()}")

    # Parse arguments
    parser = argparse.ArgumentParser(description='QAOA for QS/QV Prediction (32Q)')
    parser.add_argument('--data', type=str, default='data/c5_Matrix.csv',
                        help='Path to dataset')
    parser.add_argument('--qubits', type=int, default=32,
                        help='Number of qubits (max 33 for single H200)')
    parser.add_argument('--layers', type=int, default=6,
                        help='Number of QAOA layers')
    parser.add_argument('--shots', type=int, default=8192,
                        help='Number of measurement shots')
    parser.add_argument('--iterations', type=int, default=300,
                        help='Max optimization iterations')
    parser.add_argument('--optimizer', type=str, default='COBYLA',
                        choices=['COBYLA', 'SPSA', 'BFGS'],
                        help='Optimizer to use')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='lightning.gpu',
                        help='PennyLane device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    n_qubits = min(args.qubits, 33)  # Cap at 33 for single H200

    print(f"\nConfiguration:")
    print(f"  Qubits: {n_qubits} (reduced from 39)")
    print(f"  Layers: {args.layers}")
    print(f"  Shots: {args.shots}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Device: {args.device}")

    # Calculate memory requirement
    mem_gb = (2 ** n_qubits * 16) / (1024**3)
    print(f"  State vector size: {mem_gb:.1f} GB")

    # Step 1: Load dataset
    print("\n" + "-" * 70)
    print("STEP 1: Loading dataset")
    print("-" * 70)

    events, stats = load_dataset(args.data)

    # Step 2: Select top columns
    print("\n" + "-" * 70)
    print(f"STEP 2: Selecting top {n_qubits} columns")
    print("-" * 70)

    selected_cols = select_top_columns(stats, n_qubits)
    print(f"Selected column indices: {selected_cols}")
    print(f"Selected columns: {[f'QV_{i+1}' for i in selected_cols]}")

    # Create reduced dataset
    events_reduced = events[:, selected_cols]

    # Recompute statistics for reduced columns
    column_frequencies = events_reduced.mean(axis=0)
    cooccurrence = np.dot(events_reduced.T, events_reduced) / len(events_reduced)
    np.fill_diagonal(cooccurrence, 0)

    stats_reduced = {
        'n_events': stats['n_events'],
        'n_columns': n_qubits,
        'column_frequencies': column_frequencies,
        'cooccurrence': cooccurrence,
        'sparsity': 1 - column_frequencies.mean(),
        'dataset_hash': stats['dataset_hash'],
        'original_indices': selected_cols
    }

    print(f"Reduced frequency range: {column_frequencies.min():.4f} - {column_frequencies.max():.4f}")

    # Step 3: Compute Hamiltonian weights
    print("\n" + "-" * 70)
    print("STEP 3: Computing Hamiltonian weights")
    print("-" * 70)

    h, J = compute_hamiltonian_weights(stats_reduced, w_freq=1.0, w_cooc=0.5)

    # Step 4: Create QAOA circuit
    print("\n" + "-" * 70)
    print("STEP 4: Creating QAOA circuit")
    print("-" * 70)

    # Try GPU first, fall back to lightning.qubit (optimized CPU)
    device_name = args.device
    try:
        import pennylane as qml
        test_dev = qml.device(device_name, wires=4)
        print(f"Using device: {device_name}")
    except Exception as e:
        print(f"Failed to create {device_name}: {e}")
        device_name = 'lightning.qubit'
        print(f"Falling back to: {device_name}")

    qaoa_qnode = create_qaoa_qnode(
        n_qubits=n_qubits,
        n_layers=args.layers,
        h=h,
        J=J,
        mixer_type='xy',
        device_name=device_name,
        n_shots=args.shots
    )

    # Step 5: Initialize parameters
    print("\n" + "-" * 70)
    print("STEP 5: Initializing QAOA parameters")
    print("-" * 70)

    initial_params = get_initial_params(args.layers, seed=args.seed)
    print(f"Initial parameters shape: {initial_params.shape}")

    # Step 6: Run optimization
    print("\n" + "-" * 70)
    print("STEP 6: Running QAOA optimization")
    print("-" * 70)

    optimization_results = run_optimization(
        qaoa_qnode,
        initial_params,
        optimizer=args.optimizer,
        max_iterations=args.iterations,
        seed=args.seed
    )

    # Step 7: Sample from optimized circuit
    print("\n" + "-" * 70)
    print("STEP 7: Sampling from optimized circuit")
    print("-" * 70)

    sampling_qnode = create_sampling_qnode(
        n_qubits=n_qubits,
        n_layers=args.layers,
        h=h,
        J=J,
        mixer_type='xy',
        device_name=device_name,
        n_shots=args.shots
    )

    optimal_params = optimization_results['optimal_params']
    samples = sampling_qnode(optimal_params)
    print(f"Collected {len(samples)} samples")

    # Step 8: Process predictions
    print("\n" + "-" * 70)
    print("STEP 8: Processing predictions")
    print("-" * 70)

    # Get predictions in reduced space
    n_select = min(20, n_qubits)
    predictions_reduced, analysis = samples_to_predictions(samples, n_select=n_select)

    # Map back to original column indices
    predictions_original = selected_cols[predictions_reduced]

    print(f"\nPredicted columns (original indices):")
    for rank, (red_idx, orig_idx) in enumerate(zip(predictions_reduced, predictions_original), 1):
        freq = analysis['column_frequencies'][red_idx]
        print(f"  {rank:2d}. QV_{orig_idx+1:2d} (reduced idx {red_idx}, freq: {freq:.4f})")

    # Step 9: Save results
    print("\n" + "-" * 70)
    print("STEP 9: Saving results")
    print("-" * 70)

    os.makedirs(args.output, exist_ok=True)

    # Create config for provenance
    config = QAOAConfig(
        data_path=args.data,
        n_qubits=n_qubits,
        n_layers=args.layers,
        n_shots=args.shots,
        max_iterations=args.iterations,
        optimizer=args.optimizer,
        seed=args.seed
    )

    # Generate provenance
    provenance = generate_provenance(
        config,
        stats_reduced,
        optimization_results,
        predictions_original,
        analysis
    )

    # Add reduced column info
    provenance['column_reduction'] = {
        'original_columns': 39,
        'reduced_columns': n_qubits,
        'selected_indices': selected_cols.tolist(),
        'selected_names': [f'QV_{i+1}' for i in selected_cols]
    }

    results_path = save_results(
        provenance,
        optimization_results,
        analysis,
        output_dir=args.output
    )

    # Final summary
    print("\n" + "=" * 70)
    print("QAOA EXECUTION COMPLETE (32Q VERSION)")
    print("=" * 70)
    print(f"\nTop {n_select} Predicted Least Likely QV Columns:")
    print(f"  {', '.join([f'QV_{i+1}' for i in predictions_original])}")
    print(f"\nProvenance saved to: {results_path}")
    print(f"Result hash: {provenance['result_hash_sha512'][:64]}...")
    print(f"Execution ended: {datetime.now().isoformat()}")

    return provenance


if __name__ == '__main__':
    main()
