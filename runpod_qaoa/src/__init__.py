"""
QAOA for QS/QV Prediction
Dr. Jack Hammer - QMLE Director

Quantum Approximate Optimization Algorithm implementation
for predicting least-likely QV column activations.
"""

from .qaoa_config import QAOAConfig, PennyLaneConfig, QAOA_CONFIG, PENNYLANE_CONFIG
from .data_loader import load_dataset, prepare_qaoa_data, compute_hamiltonian_weights
from .qaoa_circuit import (
    create_cost_hamiltonian,
    create_qaoa_circuit,
    create_qaoa_qnode,
    create_sampling_qnode,
    get_initial_params
)
from .optimizer import QAOAOptimizer, run_optimization
from .utils import (
    samples_to_predictions,
    evaluate_prediction,
    generate_provenance,
    save_results,
    print_prediction_summary
)

__all__ = [
    'QAOAConfig',
    'PennyLaneConfig',
    'QAOA_CONFIG',
    'PENNYLANE_CONFIG',
    'load_dataset',
    'prepare_qaoa_data',
    'compute_hamiltonian_weights',
    'create_cost_hamiltonian',
    'create_qaoa_circuit',
    'create_qaoa_qnode',
    'create_sampling_qnode',
    'get_initial_params',
    'QAOAOptimizer',
    'run_optimization',
    'samples_to_predictions',
    'evaluate_prediction',
    'generate_provenance',
    'save_results',
    'print_prediction_summary'
]

__version__ = '1.0.0'
__author__ = 'Dr. Jack Hammer (QMLE Director)'
