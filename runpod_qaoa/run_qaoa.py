#!/usr/bin/env python3
"""
QAOA Main Execution Script for QS/QV Prediction
Dr. Jack Hammer - QMLE Director

RunPod 2x H200 GPU Configuration
PennyLane + cuQuantum Integration
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import (
    QAOAConfig,
    QAOA_CONFIG,
    PENNYLANE_CONFIG,
    load_dataset,
    prepare_qaoa_data,
    create_qaoa_qnode,
    create_sampling_qnode,
    get_initial_params,
    run_optimization,
    samples_to_predictions,
    evaluate_prediction,
    generate_provenance,
    save_results,
    print_prediction_summary
)


def main():
    """Main QAOA execution pipeline."""

    print("=" * 70)
    print("QAOA FOR QS/QV PREDICTION")
    print("Dr. Jack Hammer - QMLE Director")
    print("=" * 70)
    print(f"Execution started: {datetime.now().isoformat()}")

    # Parse arguments
    parser = argparse.ArgumentParser(description='QAOA for QS/QV Prediction')
    parser.add_argument('--data', type=str, default='data/c5_Matrix.csv',
                        help='Path to dataset')
    parser.add_argument('--layers', type=int, default=8,
                        help='Number of QAOA layers')
    parser.add_argument('--shots', type=int, default=8192,
                        help='Number of measurement shots')
    parser.add_argument('--iterations', type=int, default=500,
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

    # Create configuration
    config = QAOAConfig(
        data_path=args.data,
        n_layers=args.layers,
        n_shots=args.shots,
        max_iterations=args.iterations,
        optimizer=args.optimizer,
        seed=args.seed
    )

    print(f"\nConfiguration:")
    print(f"  Qubits: {config.n_qubits}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Mixer: {config.mixer_type}")
    print(f"  Shots: {config.n_shots}")
    print(f"  Optimizer: {config.optimizer}")
    print(f"  Device: {args.device}")

    # Step 1: Load and prepare data
    print("\n" + "-" * 70)
    print("STEP 1: Loading and preparing data")
    print("-" * 70)

    qaoa_data = prepare_qaoa_data(config)

    # Step 2: Create QAOA circuit
    print("\n" + "-" * 70)
    print("STEP 2: Creating QAOA circuit")
    print("-" * 70)

    qaoa_qnode = create_qaoa_qnode(
        n_qubits=config.n_qubits,
        n_layers=config.n_layers,
        h=qaoa_data['h'],
        J=qaoa_data['J'],
        mixer_type=config.mixer_type,
        device_name=args.device,
        n_shots=config.n_shots
    )

    # Step 3: Initialize parameters
    print("\n" + "-" * 70)
    print("STEP 3: Initializing QAOA parameters")
    print("-" * 70)

    initial_params = get_initial_params(config.n_layers, seed=config.seed)
    print(f"Initial parameters shape: {initial_params.shape}")
    print(f"Gammas: {initial_params[:config.n_layers]}")
    print(f"Betas: {initial_params[config.n_layers:]}")

    # Step 4: Run optimization
    print("\n" + "-" * 70)
    print("STEP 4: Running QAOA optimization")
    print("-" * 70)

    optimization_results = run_optimization(
        qaoa_qnode,
        initial_params,
        optimizer=config.optimizer,
        max_iterations=config.max_iterations,
        seed=config.seed
    )

    # Step 5: Sample from optimized circuit
    print("\n" + "-" * 70)
    print("STEP 5: Sampling from optimized circuit")
    print("-" * 70)

    sampling_qnode = create_sampling_qnode(
        n_qubits=config.n_qubits,
        n_layers=config.n_layers,
        h=qaoa_data['h'],
        J=qaoa_data['J'],
        mixer_type=config.mixer_type,
        device_name=args.device,
        n_shots=config.n_shots
    )

    optimal_params = optimization_results['optimal_params']
    samples = sampling_qnode(optimal_params)
    print(f"Collected {len(samples)} samples")

    # Step 6: Process predictions
    print("\n" + "-" * 70)
    print("STEP 6: Processing predictions")
    print("-" * 70)

    predictions, analysis = samples_to_predictions(samples, n_select=config.n_select)

    # Step 7: Evaluate predictions
    print("\n" + "-" * 70)
    print("STEP 7: Evaluating predictions")
    print("-" * 70)

    evaluation = evaluate_prediction(
        predictions,
        baseline_ranking=qaoa_data['baseline_ranking']
    )

    # Print summary
    print_prediction_summary(predictions, analysis, evaluation)

    # Step 8: Generate provenance and save results
    print("\n" + "-" * 70)
    print("STEP 8: Saving results")
    print("-" * 70)

    provenance = generate_provenance(
        config,
        qaoa_data['stats'],
        optimization_results,
        predictions,
        analysis
    )

    results_path = save_results(
        provenance,
        optimization_results,
        analysis,
        output_dir=args.output
    )

    # Final summary
    print("\n" + "=" * 70)
    print("QAOA EXECUTION COMPLETE")
    print("=" * 70)
    print(f"Provenance saved to: {results_path}")
    print(f"Result hash: {provenance['result_hash_sha512'][:64]}...")
    print(f"Execution ended: {datetime.now().isoformat()}")

    return provenance


if __name__ == '__main__':
    main()
