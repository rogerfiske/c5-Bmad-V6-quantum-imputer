"""
QAOA Optimizer
Dr. Jack Hammer - QMLE Director

Classical optimization loop for QAOA parameters.
Supports multiple optimizers: COBYLA, SPSA, Adam.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Callable, Dict, List, Tuple, Optional
import time
from datetime import datetime


class QAOAOptimizer:
    """
    Optimizer for QAOA parameters.

    Wraps scipy and custom optimizers with logging and early stopping.
    """

    def __init__(
        self,
        cost_fn: Callable,
        n_params: int,
        optimizer: str = 'COBYLA',
        max_iterations: int = 500,
        convergence_tol: float = 1e-6,
        seed: int = 42
    ):
        """
        Initialize optimizer.

        Args:
            cost_fn: Function that takes params and returns cost (to minimize)
            n_params: Number of parameters
            optimizer: 'COBYLA', 'SPSA', 'BFGS', or 'Adam'
            max_iterations: Maximum optimization iterations
            convergence_tol: Convergence tolerance
            seed: Random seed for SPSA
        """
        self.cost_fn = cost_fn
        self.n_params = n_params
        self.optimizer = optimizer.upper()
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.seed = seed

        # Logging
        self.history = {
            'costs': [],
            'params': [],
            'timestamps': [],
            'iterations': 0
        }

        # Best found
        self.best_cost = float('inf')
        self.best_params = None

        np.random.seed(seed)

    def _callback(self, params):
        """Callback for logging during optimization."""
        cost = self.cost_fn(params)
        self.history['costs'].append(cost)
        self.history['params'].append(params.copy())
        self.history['timestamps'].append(datetime.now().isoformat())
        self.history['iterations'] += 1

        if cost < self.best_cost:
            self.best_cost = cost
            self.best_params = params.copy()

        # Print progress every 10 iterations
        if self.history['iterations'] % 10 == 0:
            print(f"  Iteration {self.history['iterations']}: cost = {cost:.6f} (best = {self.best_cost:.6f})")

    def optimize(self, initial_params: np.ndarray) -> Dict:
        """
        Run optimization.

        Args:
            initial_params: Starting parameters

        Returns:
            Dictionary with optimization results
        """
        print(f"\nStarting {self.optimizer} optimization...")
        print(f"  Initial cost: {self.cost_fn(initial_params):.6f}")
        start_time = time.time()

        if self.optimizer == 'COBYLA':
            result = self._optimize_cobyla(initial_params)
        elif self.optimizer == 'SPSA':
            result = self._optimize_spsa(initial_params)
        elif self.optimizer == 'BFGS':
            result = self._optimize_bfgs(initial_params)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        elapsed = time.time() - start_time

        print(f"\nOptimization complete!")
        print(f"  Final cost: {self.best_cost:.6f}")
        print(f"  Total iterations: {self.history['iterations']}")
        print(f"  Time elapsed: {elapsed:.1f}s")

        return {
            'optimal_params': self.best_params,
            'optimal_cost': self.best_cost,
            'history': self.history,
            'elapsed_time': elapsed,
            'success': result.success if hasattr(result, 'success') else True
        }

    def _optimize_cobyla(self, initial_params: np.ndarray):
        """COBYLA optimizer (derivative-free, constraint-capable)."""
        result = minimize(
            self.cost_fn,
            initial_params,
            method='COBYLA',
            options={
                'maxiter': self.max_iterations,
                'rhobeg': 0.5,
                'tol': self.convergence_tol
            },
            callback=self._callback
        )
        return result

    def _optimize_bfgs(self, initial_params: np.ndarray):
        """BFGS optimizer (requires gradients, uses finite differences)."""
        result = minimize(
            self.cost_fn,
            initial_params,
            method='BFGS',
            options={
                'maxiter': self.max_iterations,
                'gtol': self.convergence_tol
            },
            callback=self._callback
        )
        return result

    def _optimize_spsa(self, initial_params: np.ndarray):
        """
        SPSA optimizer (Simultaneous Perturbation Stochastic Approximation).

        Noise-robust, good for noisy quantum cost functions.
        """
        params = initial_params.copy()

        # SPSA hyperparameters
        a = 0.1
        c = 0.1
        A = self.max_iterations * 0.1
        alpha = 0.602
        gamma_spsa = 0.101

        for k in range(self.max_iterations):
            # Gain sequences
            ak = a / ((k + 1 + A) ** alpha)
            ck = c / ((k + 1) ** gamma_spsa)

            # Random perturbation direction
            delta = 2 * (np.random.randint(0, 2, self.n_params) - 0.5)

            # Perturbed evaluations
            params_plus = params + ck * delta
            params_minus = params - ck * delta

            cost_plus = self.cost_fn(params_plus)
            cost_minus = self.cost_fn(params_minus)

            # Gradient estimate
            gradient = (cost_plus - cost_minus) / (2 * ck * delta + 1e-10)

            # Update
            params = params - ak * gradient

            # Callback
            self._callback(params)

            # Early stopping
            if len(self.history['costs']) > 20:
                recent = self.history['costs'][-20:]
                if max(recent) - min(recent) < self.convergence_tol:
                    print(f"  Converged at iteration {k}")
                    break

        class Result:
            success = True
            x = params

        return Result()


def run_optimization(
    qaoa_qnode,
    initial_params: np.ndarray,
    optimizer: str = 'COBYLA',
    max_iterations: int = 500,
    seed: int = 42
) -> Dict:
    """
    Run QAOA optimization.

    Args:
        qaoa_qnode: PennyLane QNode returning cost expectation
        initial_params: Initial parameter values
        optimizer: Optimizer name
        max_iterations: Max iterations
        seed: Random seed

    Returns:
        Optimization results dictionary
    """
    # Negate because QAOA maximizes but optimizers minimize
    def cost_fn(params):
        return -qaoa_qnode(params)

    opt = QAOAOptimizer(
        cost_fn=cost_fn,
        n_params=len(initial_params),
        optimizer=optimizer,
        max_iterations=max_iterations,
        seed=seed
    )

    return opt.optimize(initial_params)
