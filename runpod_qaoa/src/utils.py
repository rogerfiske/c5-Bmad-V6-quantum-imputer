"""
Utility Functions for QAOA
Dr. Jack Hammer - QMLE Director

Post-processing, analysis, and visualization utilities.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
from hashlib import sha512
import json
from datetime import datetime


def samples_to_predictions(
    samples: np.ndarray,
    n_select: int = 20
) -> Tuple[np.ndarray, Dict]:
    """
    Convert QAOA measurement samples to column predictions.

    For each sample bitstring, identify which columns have |1⟩ (activated).
    Aggregate across samples to find most frequently selected columns.

    Args:
        samples: Array of shape (n_shots, n_qubits) with 0/1 values
        n_select: Number of columns to predict

    Returns:
        predicted_columns: Array of n_select column indices (0-indexed)
        analysis: Dictionary with detailed sample analysis
    """
    n_shots, n_qubits = samples.shape

    # Count how often each column appears as |1⟩ across samples
    column_counts = samples.sum(axis=0)
    column_frequencies = column_counts / n_shots

    # Sort by frequency descending (most commonly selected in |1⟩ state)
    # These represent the MAXIMIZERS of the cost function
    # For "least likely" prediction, these are our targets
    ranking = np.argsort(-column_frequencies)
    predicted_columns = ranking[:n_select]

    # Analyze Hamming weights of samples
    hamming_weights = samples.sum(axis=1)
    hw_counter = Counter(hamming_weights.astype(int))

    # Find most common bitstrings
    bitstring_counts = Counter(tuple(row) for row in samples.astype(int))
    top_bitstrings = bitstring_counts.most_common(10)

    analysis = {
        'n_shots': n_shots,
        'column_frequencies': column_frequencies,
        'hamming_weight_distribution': dict(hw_counter),
        'mean_hamming_weight': hamming_weights.mean(),
        'top_bitstrings': top_bitstrings,
        'predicted_column_frequencies': column_frequencies[predicted_columns]
    }

    return predicted_columns, analysis


def evaluate_prediction(
    predicted_columns: np.ndarray,
    actual_activations: np.ndarray = None,
    baseline_ranking: np.ndarray = None
) -> Dict:
    """
    Evaluate QAOA prediction quality.

    Args:
        predicted_columns: Array of predicted column indices (0-indexed)
        actual_activations: Ground truth for event 11624 (if known)
        baseline_ranking: Baseline 1 ranking for comparison

    Returns:
        Evaluation metrics dictionary
    """
    n_predicted = len(predicted_columns)

    evaluation = {
        'predicted_columns': predicted_columns.tolist(),
        'predicted_column_names': [f"QV_{i+1}" for i in predicted_columns],
        'n_predicted': n_predicted
    }

    if baseline_ranking is not None:
        # Compare to baseline
        overlap_with_baseline = len(
            set(predicted_columns) & set(baseline_ranking[:n_predicted])
        )
        evaluation['overlap_with_baseline'] = overlap_with_baseline
        evaluation['baseline_agreement_pct'] = overlap_with_baseline / n_predicted

        # Find differences
        in_qaoa_not_baseline = set(predicted_columns) - set(baseline_ranking[:n_predicted])
        in_baseline_not_qaoa = set(baseline_ranking[:n_predicted]) - set(predicted_columns)

        evaluation['unique_to_qaoa'] = [f"QV_{i+1}" for i in in_qaoa_not_baseline]
        evaluation['unique_to_baseline'] = [f"QV_{i+1}" for i in in_baseline_not_qaoa]

    if actual_activations is not None:
        # Ground truth evaluation
        actual_active = set(np.where(actual_activations == 1)[0])
        overlap_with_actual = len(set(predicted_columns) & actual_active)
        evaluation['overlap_with_actual'] = overlap_with_actual
        evaluation['success_metric'] = 'EXCELLENT' if overlap_with_actual in [0, 5] else \
                                       'GOOD' if overlap_with_actual in [1, 4] else 'POOR'

    return evaluation


def generate_provenance(
    config,
    data_stats: Dict,
    optimization_results: Dict,
    predictions: np.ndarray,
    analysis: Dict
) -> Dict:
    """
    Generate full provenance record for the QAOA experiment.

    Args:
        config: QAOAConfig instance
        data_stats: Dataset statistics
        optimization_results: Optimization output
        predictions: Predicted columns
        analysis: Sample analysis

    Returns:
        Provenance dictionary with SHA-512 hashes
    """
    timestamp = datetime.now().isoformat()

    provenance = {
        'experiment_id': f'QAOA-QMLE-{timestamp[:10]}',
        'timestamp': timestamp,
        'researcher': 'Dr. Jack Hammer (QMLE Director)',
        'method': 'QAOA with XY-mixer',

        'configuration': {
            'n_qubits': config.n_qubits,
            'n_layers': config.n_layers,
            'mixer_type': config.mixer_type,
            'optimizer': config.optimizer,
            'max_iterations': config.max_iterations,
            'n_shots': config.n_shots,
            'seed': config.seed,
            'w_freq': config.w_freq,
            'w_cooc': config.w_cooc
        },

        'dataset': {
            'path': config.data_path,
            'hash_sha512': data_stats['dataset_hash'],
            'n_events': data_stats['n_events'],
            'n_columns': data_stats['n_columns'],
            'sparsity': data_stats['sparsity']
        },

        'optimization': {
            'optimal_cost': optimization_results['optimal_cost'],
            'iterations': optimization_results['history']['iterations'],
            'elapsed_time': optimization_results['elapsed_time'],
            'success': optimization_results['success']
        },

        'predictions': {
            'columns': predictions.tolist(),
            'column_names': [f'QV_{i+1}' for i in predictions],
            'mean_hamming_weight': analysis['mean_hamming_weight']
        }
    }

    # Compute result hash
    result_str = json.dumps(provenance, sort_keys=True)
    provenance['result_hash_sha512'] = sha512(result_str.encode()).hexdigest()

    return provenance


def save_results(
    provenance: Dict,
    optimization_results: Dict,
    analysis: Dict,
    output_dir: str = 'results'
) -> str:
    """
    Save all results to files.

    Args:
        provenance: Provenance dictionary
        optimization_results: Optimization output
        analysis: Sample analysis
        output_dir: Output directory

    Returns:
        Path to main results file
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    timestamp = provenance['timestamp'][:19].replace(':', '-')

    # Save provenance
    provenance_path = f"{output_dir}/qaoa_provenance_{timestamp}.json"
    with open(provenance_path, 'w') as f:
        json.dump(provenance, f, indent=2)

    # Save optimization history
    history_path = f"{output_dir}/qaoa_optimization_history_{timestamp}.json"
    history = optimization_results['history'].copy()
    history['params'] = [p.tolist() for p in history['params']]
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save detailed analysis
    analysis_path = f"{output_dir}/qaoa_analysis_{timestamp}.json"
    analysis_save = {
        'hamming_weight_distribution': analysis['hamming_weight_distribution'],
        'mean_hamming_weight': float(analysis['mean_hamming_weight']),
        'column_frequencies': analysis['column_frequencies'].tolist(),
        'n_shots': analysis['n_shots']
    }
    with open(analysis_path, 'w') as f:
        json.dump(analysis_save, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - Provenance: {provenance_path}")
    print(f"  - History: {history_path}")
    print(f"  - Analysis: {analysis_path}")

    return provenance_path


def print_prediction_summary(
    predictions: np.ndarray,
    analysis: Dict,
    evaluation: Dict = None
):
    """
    Print formatted prediction summary.

    Args:
        predictions: Predicted column indices
        analysis: Sample analysis
        evaluation: Optional evaluation metrics
    """
    print("\n" + "=" * 70)
    print("QAOA PREDICTION RESULTS")
    print("=" * 70)

    print(f"\nTop {len(predictions)} Least Likely QV Columns:")
    print("-" * 70)

    for rank, col_idx in enumerate(predictions, 1):
        freq = analysis['column_frequencies'][col_idx]
        print(f"  {rank:2d}. QV_{col_idx+1:2d}  (QAOA selection frequency: {freq:.4f})")

    print(f"\nSample Statistics:")
    print(f"  Mean Hamming weight: {analysis['mean_hamming_weight']:.2f}")
    print(f"  Hamming weight distribution: {analysis['hamming_weight_distribution']}")

    if evaluation:
        print(f"\nComparison with Baseline 1:")
        print(f"  Overlap: {evaluation.get('overlap_with_baseline', 'N/A')} / {len(predictions)}")
        print(f"  Agreement: {evaluation.get('baseline_agreement_pct', 'N/A'):.1%}")

        if 'unique_to_qaoa' in evaluation:
            print(f"  Unique to QAOA: {', '.join(evaluation['unique_to_qaoa'])}")
            print(f"  Unique to Baseline: {', '.join(evaluation['unique_to_baseline'])}")

    print("=" * 70)
