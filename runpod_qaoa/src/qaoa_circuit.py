"""
QAOA Circuit Implementation for QS/QV Prediction
Dr. Jack Hammer - QMLE Director

Implements QAOA with XY-mixer for Hamming weight preservation.
The XY-mixer naturally enforces the constraint that we select
a fixed number of columns (similar to the 5-activation constraint
in the original data).
"""

import pennylane as qml
import numpy as np
from typing import Tuple, Optional


def create_cost_hamiltonian(h: np.ndarray, J: np.ndarray) -> qml.Hamiltonian:
    """
    Create the cost Hamiltonian for QAOA.

    H_C = -sum_i h_i * Z_i - sum_{i<j} J_ij * Z_i * Z_j

    This encodes the optimization objective:
    - Maximize H_C to find columns that are:
      - Less frequent (high h_i)
      - Less co-occurring (negative J_ij for common pairs)

    Args:
        h: Single-qubit weights (n_qubits,)
        J: Two-qubit couplings (n_qubits, n_qubits)

    Returns:
        PennyLane Hamiltonian object
    """
    n_qubits = len(h)
    coeffs = []
    ops = []

    # Single-qubit Z terms
    for i in range(n_qubits):
        if abs(h[i]) > 1e-10:
            coeffs.append(-h[i])  # Negative for maximization
            ops.append(qml.PauliZ(i))

    # Two-qubit ZZ terms
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if abs(J[i, j]) > 1e-10:
                coeffs.append(-J[i, j])
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    return qml.Hamiltonian(coeffs, ops)


def apply_cost_layer(gamma: float, h: np.ndarray, J: np.ndarray, n_qubits: int):
    """
    Apply the cost unitary: U_C(gamma) = exp(-i * gamma * H_C)

    Decomposed into:
    - RZ rotations for single-qubit terms
    - CNOT-RZ-CNOT for ZZ terms

    Args:
        gamma: QAOA angle for cost layer
        h: Single-qubit weights
        J: Two-qubit couplings
        n_qubits: Number of qubits
    """
    # Single-qubit RZ rotations: exp(-i * gamma * h_i * Z_i)
    for i in range(n_qubits):
        if abs(h[i]) > 1e-10:
            qml.RZ(2 * gamma * h[i], wires=i)

    # Two-qubit ZZ rotations: exp(-i * gamma * J_ij * Z_i * Z_j)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if abs(J[i, j]) > 1e-10:
                # ZZ rotation via CNOT-RZ-CNOT
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * gamma * J[i, j], wires=j)
                qml.CNOT(wires=[i, j])


def apply_xy_mixer(beta: float, n_qubits: int):
    """
    Apply the XY-mixer: U_B(beta) = exp(-i * beta * H_B)

    H_B = sum_{i<j} (X_i X_j + Y_i Y_j)

    The XY-mixer preserves Hamming weight (number of |1⟩ states),
    which is crucial for our constraint-preserving optimization.

    We use nearest-neighbor XY interactions for efficiency,
    but can extend to all-pairs for stronger mixing.

    Args:
        beta: QAOA angle for mixer layer
        n_qubits: Number of qubits
    """
    # Apply XY interactions in two sweeps (even-odd pattern for parallelism)

    # First sweep: even pairs (0-1, 2-3, 4-5, ...)
    for i in range(0, n_qubits - 1, 2):
        apply_xy_gate(beta, i, i + 1)

    # Second sweep: odd pairs (1-2, 3-4, 5-6, ...)
    for i in range(1, n_qubits - 1, 2):
        apply_xy_gate(beta, i, i + 1)

    # Wrap-around connection for ring topology (helps mixing)
    if n_qubits > 2:
        apply_xy_gate(beta, n_qubits - 1, 0)


def apply_xy_gate(theta: float, wire1: int, wire2: int):
    """
    Apply XY gate: exp(-i * theta * (XX + YY) / 2)

    This can be decomposed as:
    CNOT(1,2) - RY(1, theta) - CNOT(1,2)

    Or more efficiently using native gates.

    Args:
        theta: Rotation angle
        wire1, wire2: Qubit indices
    """
    # Efficient decomposition of XY interaction
    # exp(-i * theta * (XX + YY)) preserves Hamming weight

    qml.CNOT(wires=[wire1, wire2])
    qml.RY(theta, wires=wire1)
    qml.RX(theta, wires=wire2)
    qml.CNOT(wires=[wire1, wire2])


def apply_x_mixer(beta: float, n_qubits: int):
    """
    Standard X-mixer (does NOT preserve Hamming weight).

    H_B = sum_i X_i

    Use this for comparison or when constraint preservation
    is not required.

    Args:
        beta: QAOA angle for mixer layer
        n_qubits: Number of qubits
    """
    for i in range(n_qubits):
        qml.RX(2 * beta, wires=i)


def create_qaoa_circuit(
    n_qubits: int,
    n_layers: int,
    h: np.ndarray,
    J: np.ndarray,
    mixer_type: str = 'xy',
    initial_state: Optional[np.ndarray] = None
):
    """
    Create the full QAOA circuit.

    |psi(gamma, beta)> = U_B(beta_p) U_C(gamma_p) ... U_B(beta_1) U_C(gamma_1) |+>

    Args:
        n_qubits: Number of qubits
        n_layers: Number of QAOA layers (p)
        h: Single-qubit Hamiltonian weights
        J: Two-qubit Hamiltonian couplings
        mixer_type: 'xy' or 'x'
        initial_state: Optional initial bitstring for warm-start

    Returns:
        Function that builds the circuit given parameters
    """

    def circuit(params):
        """
        Build QAOA circuit with given parameters.

        Args:
            params: Array of shape (2 * n_layers,)
                    [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
        """
        gammas = params[:n_layers]
        betas = params[n_layers:]

        # Initial state: |+>^n (equal superposition)
        # Or warm-start from baseline
        if initial_state is not None:
            # Prepare specific initial state
            for i, bit in enumerate(initial_state):
                if bit == 1:
                    qml.PauliX(wires=i)
        else:
            # Standard: Hadamard on all qubits
            for i in range(n_qubits):
                qml.Hadamard(wires=i)

        # Apply QAOA layers
        for layer in range(n_layers):
            # Cost layer
            apply_cost_layer(gammas[layer], h, J, n_qubits)

            # Mixer layer
            if mixer_type == 'xy':
                apply_xy_mixer(betas[layer], n_qubits)
            else:
                apply_x_mixer(betas[layer], n_qubits)

    return circuit


def create_qaoa_qnode(
    n_qubits: int,
    n_layers: int,
    h: np.ndarray,
    J: np.ndarray,
    mixer_type: str = 'xy',
    device_name: str = 'lightning.gpu',
    n_shots: int = 8192
):
    """
    Create a PennyLane QNode for QAOA.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of QAOA layers
        h: Hamiltonian single-qubit weights
        J: Hamiltonian two-qubit couplings
        mixer_type: 'xy' or 'x'
        device_name: PennyLane device
        n_shots: Number of measurement shots

    Returns:
        QNode that returns expectation value of cost Hamiltonian
    """
    # Create device
    try:
        dev = qml.device(device_name, wires=n_qubits)
        print(f"Using device: {device_name}")
    except Exception as e:
        print(f"Failed to create {device_name}, falling back to default.qubit")
        dev = qml.device('default.qubit', wires=n_qubits)

    # Create cost Hamiltonian
    cost_hamiltonian = create_cost_hamiltonian(h, J)

    # Create circuit function
    circuit_fn = create_qaoa_circuit(n_qubits, n_layers, h, J, mixer_type)

    @qml.qnode(dev, diff_method='best')
    def qaoa_expectation(params):
        """Compute expectation value of cost Hamiltonian."""
        circuit_fn(params)
        return qml.expval(cost_hamiltonian)

    return qaoa_expectation


def create_sampling_qnode(
    n_qubits: int,
    n_layers: int,
    h: np.ndarray,
    J: np.ndarray,
    mixer_type: str = 'xy',
    device_name: str = 'lightning.gpu',
    n_shots: int = 8192
):
    """
    Create a QNode for sampling measurement outcomes.

    Args:
        Same as create_qaoa_qnode

    Returns:
        QNode that returns measurement samples
    """
    try:
        dev = qml.device(device_name, wires=n_qubits, shots=n_shots)
    except:
        dev = qml.device('default.qubit', wires=n_qubits, shots=n_shots)

    circuit_fn = create_qaoa_circuit(n_qubits, n_layers, h, J, mixer_type)

    @qml.qnode(dev)
    def qaoa_sample(params):
        """Sample from QAOA output distribution."""
        circuit_fn(params)
        return qml.sample()

    return qaoa_sample


def get_initial_params(n_layers: int, seed: int = 42) -> np.ndarray:
    """
    Initialize QAOA parameters.

    Following QAOA best practices:
    - gamma initialized near 0 (weak cost evolution initially)
    - beta initialized near pi/4 (moderate mixing)

    Args:
        n_layers: Number of QAOA layers
        seed: Random seed

    Returns:
        Initial parameters array (2 * n_layers,)
    """
    np.random.seed(seed)

    # Initialize gammas small, betas around pi/4
    gammas = 0.1 * np.random.randn(n_layers)
    betas = np.pi / 4 + 0.1 * np.random.randn(n_layers)

    return np.concatenate([gammas, betas])
