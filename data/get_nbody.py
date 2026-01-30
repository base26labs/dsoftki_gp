"""
Simple N-body Gravitational Hamiltonian System

Generates dataset with true gradient observations for GP regression.

The Hamiltonian is:
    H(q, p) = T(p) + V(q)
    T(p) = sum_i |p_i|^2 / (2 * m_i)  (kinetic energy)
    V(q) = -G * sum_{i<j} m_i * m_j / |q_i - q_j|  (gravitational potential)

Gradients:
    dH/dp = p / m  (velocity)
    dH/dq = -dV/dq (force)

State: x = [q, p] where q = positions, p = momenta
Gradient: dx/dt = [dH/dp, -dH/dq]
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import argparse


def hamiltonian(q, p, masses, G=1.0, softening=0.1):
    """
    Compute Hamiltonian H = T + V

    Args:
        q: positions, shape (n_particles, n_dims)
        p: momenta, shape (n_particles, n_dims)
        masses: particle masses, shape (n_particles,)
        G: gravitational constant
        softening: softening parameter to prevent singularities

    Returns:
        H: scalar energy
    """
    n_particles = len(masses)

    # Kinetic energy: T = sum_i |p_i|^2 / (2 * m_i)
    T = 0.5 * np.sum(p**2 / masses[:, None])

    # Potential energy: V = -G * sum_{i<j} m_i * m_j / sqrt(r^2 + eps^2)
    # Using Plummer softening: prevents singularities at r=0
    V = 0.0
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            r = np.linalg.norm(q[i] - q[j])
            # Plummer softening: 1/sqrt(r^2 + ε^2) instead of 1/r
            V -= G * masses[i] * masses[j] / np.sqrt(r**2 + softening**2)

    return T + V


def hamiltonian_gradients(q, p, masses, G=1.0, softening=0.1):
    """
    Compute gradients of Hamiltonian.

    Returns:
        dH_dq: gradient w.r.t. positions
        dH_dp: gradient w.r.t. momenta (velocity)
    """
    n_particles = len(masses)
    n_dims = q.shape[1]

    # dH/dp = p / m (velocity)
    dH_dp = p / masses[:, None]

    # dH/dq = dV/dq with Plummer softening
    # V = -G * m_i * m_j / sqrt(r^2 + ε^2)
    # dV/dq_i = -G * m_i * m_j * d/dq_i[1/sqrt(r^2 + ε^2)]
    #         = -G * m_i * m_j * (-(q_i - q_j)) / (r^2 + ε^2)^(3/2)
    #         = G * m_i * m_j * (q_i - q_j) / (r^2 + ε^2)^(3/2)
    dH_dq = np.zeros_like(q)
    for i in range(n_particles):
        for j in range(n_particles):
            if i != j:
                r_vec = q[i] - q[j]
                r = np.linalg.norm(r_vec)
                r_soft = np.sqrt(r**2 + softening**2)
                # Gradient with Plummer softening
                dH_dq[i] += G * masses[i] * masses[j] * r_vec / (r_soft**3)

    return dH_dq, dH_dp


def hamiltonian_dynamics(t, state, masses, G=1.0, softening=0.1):
    """
    Right-hand side of Hamilton's equations for ODE solver.

    dq/dt = dH/dp
    dp/dt = -dH/dq
    """
    n_particles = len(masses)
    n_dims = state.shape[0] // (2 * n_particles)

    q = state[:n_particles * n_dims].reshape(n_particles, n_dims)
    p = state[n_particles * n_dims:].reshape(n_particles, n_dims)

    dH_dq, dH_dp = hamiltonian_gradients(q, p, masses, G, softening)

    dq_dt = dH_dp      # = dH/dp
    dp_dt = -dH_dq     # = -dH/dq

    return np.concatenate([dq_dt.flatten(), dp_dt.flatten()])


def generate_nbody_dataset(
    n_particles=10,
    n_dims=3,
    n_samples=10000,
    n_trajectories=100,
    steps_per_trajectory=100,
    dt=0.01,
    G=1.0,
    mass_range=(0.5, 2.0),
    position_scale=2.0,
    momentum_scale=0.5,
    softening=0.1,
    seed=42
):
    """
    Generate N-body Hamiltonian dataset with true gradients.

    Args:
        n_particles: number of particles
        n_dims: spatial dimensions (2 or 3)
        n_samples: total number of samples to generate
        n_trajectories: number of separate trajectories
        steps_per_trajectory: steps per trajectory
        dt: time step
        G: gravitational constant
        mass_range: (min, max) for particle masses
        position_scale: scale for initial positions
        momentum_scale: scale for initial momenta
        softening: Plummer softening parameter (prevents singularities)
        seed: random seed

    Returns:
        X: states, shape (n_samples, 2 * n_particles * n_dims)
        y: Hamiltonian values, shape (n_samples,)
        dy: gradients of H w.r.t. state, shape (n_samples, 2 * n_particles * n_dims)
    """
    np.random.seed(seed)
    
    state_dim = 2 * n_particles * n_dims  # positions + momenta
    
    X_list = []
    y_list = []
    dy_list = []
    
    samples_per_traj = n_samples // n_trajectories

    for traj_idx in tqdm(range(n_trajectories), desc="Generating trajectories"):
        # Random masses for this trajectory
        masses = np.random.uniform(mass_range[0], mass_range[1], n_particles)

        # Random initial conditions
        q0 = np.random.randn(n_particles, n_dims) * position_scale
        p0 = np.random.randn(n_particles, n_dims) * momentum_scale * masses[:, None]

        # Center of mass correction (optional, for stability)
        q0 -= np.average(q0, weights=masses, axis=0)
        p0 -= np.average(p0, weights=masses, axis=0)

        state0 = np.concatenate([q0.flatten(), p0.flatten()])

        # Integrate trajectory
        t_span = (0, steps_per_trajectory * dt)
        t_eval = np.linspace(0, t_span[1], steps_per_trajectory)

        sol = solve_ivp(
            hamiltonian_dynamics,
            t_span,
            state0,
            args=(masses, G, softening),
            t_eval=t_eval,
            method='DOP853',  # High-order method for Hamiltonian systems
            rtol=1e-10,
            atol=1e-12
        )

        # Sample points from trajectory
        indices = np.random.choice(len(sol.t), min(samples_per_traj, len(sol.t)), replace=False)

        for idx in indices:
            state = sol.y[:, idx]
            q = state[:n_particles * n_dims].reshape(n_particles, n_dims)
            p = state[n_particles * n_dims:].reshape(n_particles, n_dims)

            # Compute Hamiltonian (energy)
            H = hamiltonian(q, p, masses, G, softening)

            # Compute gradients
            dH_dq, dH_dp = hamiltonian_gradients(q, p, masses, G, softening)

            # Full gradient of H w.r.t. state [q, p]
            dH = np.concatenate([dH_dq.flatten(), dH_dp.flatten()])

            X_list.append(state)
            y_list.append(H)
            dy_list.append(dH)
    
    X = np.array(X_list)
    y = np.array(y_list)
    dy = np.array(dy_list)
    
    return X, y, dy, {'n_particles': n_particles, 'n_dims': n_dims}


if __name__ == "__main__":
    # Usage examples:
    # python get_nbody.py                              # Default: 10 particles, 3D
    # python get_nbody.py --n_particles 20 --n_dims 2  # 20 particles, 2D
    # python get_nbody.py --n_particles 5 --n_samples 5000 --percentile_filter 90

    parser = argparse.ArgumentParser(description='Generate N-body Hamiltonian dataset')
    parser.add_argument('--n_particles', type=int, default=10,
                        help='Number of particles (default: 10)')
    parser.add_argument('--n_dims', type=int, default=3,
                        help='Spatial dimensions (default: 3)')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Total number of samples (default: 10000)')
    parser.add_argument('--n_trajectories', type=int, default=100,
                        help='Number of trajectories (default: 100)')
    parser.add_argument('--steps_per_trajectory', type=int, default=200,
                        help='Steps per trajectory (default: 200)')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step (default: 0.01)')
    parser.add_argument('--G', type=float, default=1.0,
                        help='Gravitational constant (default: 1.0)')
    parser.add_argument('--softening', type=float, default=0.1,
                        help='Plummer softening parameter (default: 0.1)')
    parser.add_argument('--percentile_filter', type=float, default=95.0,
                        help='Percentile threshold for gradient filtering (default: 95.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Example: Generate dataset
    print("Generating N-body Hamiltonian dataset...")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  N particles: {args.n_particles}")
    print(f"  Spatial dims: {args.n_dims}")
    print(f"  State dimension: {2 * args.n_particles * args.n_dims}")
    print(f"  N samples: {args.n_samples}")
    print(f"  N trajectories: {args.n_trajectories}")
    print(f"  Steps per trajectory: {args.steps_per_trajectory}")
    print(f"  Time step: {args.dt}")
    print(f"  Gravitational constant: {args.G}")
    print(f"  Softening: {args.softening}")
    print(f"  Gradient filter percentile: {args.percentile_filter}")
    print(f"  Random seed: {args.seed}")
    print("=" * 50)

    X, y, dy, info = generate_nbody_dataset(
        n_particles=args.n_particles,
        n_dims=args.n_dims,
        n_samples=args.n_samples,
        n_trajectories=args.n_trajectories,
        steps_per_trajectory=args.steps_per_trajectory,
        dt=args.dt,
        G=args.G,
        softening=args.softening,
        seed=args.seed
    )

    print(f"\nDataset generated:")
    print(f"  N particles: {args.n_particles}")
    print(f"  Spatial dims: {args.n_dims}")
    print(f"  State dimension D: {X.shape[1]}")
    print(f"  N samples: {X.shape[0]}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  dy shape: {dy.shape}")

    # Filter out samples with extreme gradients
    print("\n" + "=" * 50)
    print("Filtering extreme gradients...")
    grad_norms = np.linalg.norm(dy, axis=1)

    # Use percentile-based filtering
    percentile_threshold = args.percentile_filter
    threshold = np.percentile(grad_norms, percentile_threshold)

    print(f"Gradient norm statistics BEFORE filtering:")
    print(f"  Mean:   {grad_norms.mean():.4f}")
    print(f"  Median: {np.median(grad_norms):.4f}")
    print(f"  95th percentile: {np.percentile(grad_norms, 95):.4f}")
    print(f"  99th percentile: {np.percentile(grad_norms, 99):.4f}")
    print(f"  Max:    {grad_norms.max():.4f}")

    # Filter
    keep_mask = grad_norms <= threshold
    X = X[keep_mask]
    y = y[keep_mask]
    dy = dy[keep_mask]

    n_filtered = (~keep_mask).sum()
    print(f"\nFiltered {n_filtered} samples ({100*n_filtered/len(keep_mask):.1f}%)")
    print(f"Remaining samples: {len(X)}")

    grad_norms_filtered = np.linalg.norm(dy, axis=1)
    print(f"\nGradient norm statistics AFTER filtering:")
    print(f"  Mean:   {grad_norms_filtered.mean():.4f}")
    print(f"  Median: {np.median(grad_norms_filtered):.4f}")
    print(f"  Max:    {grad_norms_filtered.max():.4f}")
    print("=" * 50)

    # Create nbody directory if it doesn't exist
    import os
    os.makedirs('nbody', exist_ok=True)

    # Create filename with n_particles and n_dims
    filename_base = f"nbody_n{args.n_particles}_d{args.n_dims}"

    # Save dataset (for torch loader)
    output_file = f'nbody/{filename_base}.npz'
    np.savez(
        output_file,
        X=X,  # Full states
        E=y,  # Full energies (Hamiltonian values)
        F=dy,  # Full gradients (forces)
        **info
    )
    print(f"Saved dataset to {output_file}")
    
    # Verify energy conservation (sanity check)
    print(f"\nEnergy statistics (should be roughly conserved per trajectory):")
    print(f"  Mean energy: {y.mean():.4f}")
    print(f"  Std energy: {y.std():.4f}")
    print(f"  Min energy: {y.min():.4f}")
    print(f"  Max energy: {y.max():.4f}")


# =============================================================================
# PyTorch Dataset Loader (similar to MD22Dataset)
# =============================================================================

class NBodyDataset(Dataset):
    """
    N-body Hamiltonian dataset loader.

    Similar to MD22Dataset but for N-body gravitational systems.
    State: x = [q, p] where q = positions, p = momenta
    Energy: Hamiltonian H(q, p)
    Forces: Gradient dH/dx (negative of Hamilton's equations RHS)
    """
    def __init__(
        self,
        npz_file="./nbody_hamiltonian_dataset.npz",
        dtype=torch.float32,
        transform=None,
        standardize=False,
        center=True,
        get_forces=False,
        unit_cube=False
    ):
        # Store arguments
        self.npz_file = npz_file
        self.dtype = dtype
        self.get_forces = get_forces
        self.transform = transform
        self.center = center
        self.standardize = standardize
        self.unit_cube = unit_cube

        # Load data
        self.raw_data = np.load(npz_file)
        self.states = torch.tensor(self.raw_data["X"], dtype=dtype)  # (N, state_dim)
        self.energies = torch.tensor(self.raw_data["E"], dtype=dtype)  # (N,)
        self.forces = torch.tensor(self.raw_data["F"], dtype=dtype)  # (N, state_dim)

        # Metadata
        self.n_particles = int(self.raw_data["n_particles"])
        self.n_dims = int(self.raw_data["n_dims"])
        self.dim = self.states.shape[1]

        print(f"Loaded N-body dataset:")
        print(f"  N particles: {self.n_particles}")
        print(f"  Spatial dims: {self.n_dims}")
        print(f"  State dim: {self.dim}")
        print(f"  N samples: {len(self.states)}")
        print(f"  Energy shape: {self.energies.shape}")
        print(f"  Force shape: {self.forces.shape}")

        # Standardization
        self.shift = 0
        self.scale = 1

        if standardize:
            if unit_cube:
                # Map to unit cube [0, 1] per dimension
                mins = self.states.min(dim=0)[0]
                maxs = self.states.max(dim=0)[0]
                ranges = maxs - mins + 1e-8  # Add epsilon to avoid division by zero

                # Store bounds for each dimension
                self.state_mins = mins
                self.state_ranges = ranges

                print(f"Unit cube normalization:")
                print(f"  Min range: {ranges.min():.4f}")
                print(f"  Max range: {ranges.max():.4f}")
                print(f"  Mean range: {ranges.mean():.4f}")

                # Normalize each dimension independently to [0, 1]
                self.states = (self.states - mins) / ranges
            else:
                # Simple scaling
                self.x_scale = 3.0
                self.states = self.states / self.x_scale

            # Standardize energies and forces together
            mu = self.energies.mean()
            sigma = torch.cat([
                (self.energies - mu).unsqueeze(-1),
                self.forces
            ], dim=-1).std()

            print(f"Energy standardization: mean={mu:.4f}, std={sigma:.4f}")
            self.energies = (self.energies - mu) / sigma

            # Normalize forces with chain rule
            # If x_norm = (x - mins) / ranges, then:
            # df/dx_norm = df/dx * dx/dx_norm = df/dx * ranges
            if unit_cube:
                self.forces = self.forces * ranges / sigma
            else:
                self.forces = self.forces / sigma

            self.shift = mu
            self.scale = sigma
        else:
            if self.center:
                self._center_energies()

    def _center_energies(self):
        """Center energies by subtracting mean."""
        self.mean = self.energies.mean()
        self.energies = self.energies - self.mean
        print(f"Centered energies: mean={self.mean:.4f}")

    def __len__(self):
        return len(self.energies)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.states[idx]
        if self.transform:
            features = self.transform(features)

        if self.get_forces:
            return features, {
                "energy": self.energies[idx],
                "neg_force": self.forces[idx],
            }
        else:
            label = self.energies[idx]
            return features, label