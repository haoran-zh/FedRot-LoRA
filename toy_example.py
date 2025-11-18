import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --- Configuration ---
NUM_ROUNDS = 30
LOCAL_STEPS = 30
LR = 0.01
CLIENT_OPTIMALS = [0.5, 1.0, 1.5]
NUM_CLIENTS = len(CLIENT_OPTIMALS)
GLOBAL_OPTIMAL = np.mean(CLIENT_OPTIMALS)
EPSILON = 1e-8  # Small value for numerical stability

# Set random seed
np.random.seed(5)


class Client:
    def __init__(self, client_id, local_optimal_dw):
        self.client_id = client_id
        self.local_optimal_dw = local_optimal_dw
        self.A = 0.0
        self.B = 0.0

    def set_AB(self, A, B):
        self.A = A
        self.B = B

    def train_local(self, num_steps, lr, train_A, train_B):
        """
        Minimize L = (B*A - local_optimal)^2
        """
        for _ in range(num_steps):
            current_dw = self.B * self.A
            error = current_dw - self.local_optimal_dw

            A_old = self.A
            B_old = self.B

            if train_B:
                # dL/dB = 2 * error * A
                grad_B = 2 * error * (A_old if train_A else self.A)
                self.B -= lr * grad_B

            if train_A:
                # dL/dA = 2 * error * B
                grad_A = 2 * error * (B_old if train_B else self.B)
                self.A -= lr * grad_A

    def align_A_and_get(self, A_ref):
        """ Rotate such that A becomes A_ref, B scales inversely. """
        if abs(self.A) < EPSILON or abs(A_ref) < EPSILON:
            return self.A, self.B
        R = A_ref / self.A
        return self.A * R, self.B / R

    def align_B_and_get(self, B_ref):
        """ Rotate such that B becomes B_ref, A scales inversely. """
        if abs(self.B) < EPSILON or abs(B_ref) < EPSILON:
            return self.A, self.B
        R = B_ref / self.B
        return self.A / R, self.B * R


class Server:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.A = 0.0
        self.B = 0.0

    def aggregate_B(self, client_Bs):
        self.B = np.mean(client_Bs)

    def aggregate_A(self, client_As):
        self.A = np.mean(client_As)

    def aggregate_AB(self, client_As, client_Bs):
        self.A = np.mean(client_As)
        self.B = np.mean(client_Bs)

    def broadcast_AB(self, clients):
        for client in clients:
            client.set_AB(self.A, self.B)

    def get_global_dw(self):
        return self.B * self.A


def run_simulation_logic():
    """
    Executes the logic for all strategies and returns the data dictionaries.
    """
    trajectory_results = {}

    # Shared Initialization
    SHARED_INITIAL_A = np.random.randn()
    SHARED_INITIAL_B = 0.0

    def init_sim_state():
        server = Server(NUM_CLIENTS)
        server.A, server.B = SHARED_INITIAL_A, SHARED_INITIAL_B
        clients = [Client(i, CLIENT_OPTIMALS[i]) for i in range(NUM_CLIENTS)]
        for client in clients:
            client.set_AB(SHARED_INITIAL_A, SHARED_INITIAL_B)
        return server, clients

    # === 1. FFA-LoRA ===
    server, clients = init_sim_state()
    traj = [(server.A, server.B)]
    frozen_A = server.A
    for _ in range(NUM_ROUNDS):
        client_Bs = []
        for client in clients:
            client.set_AB(frozen_A, server.B)
            client.train_local(LOCAL_STEPS, LR, train_A=False, train_B=True)
            client_Bs.append(client.B)
        server.aggregate_B(client_Bs)
        traj.append((frozen_A, server.B))
    trajectory_results['FFA-LoRA (Freeze A)'] = traj

    # === 2. RoLoRA ===
    server, clients = init_sim_state()
    traj = [(server.A, server.B)]
    for round in range(NUM_ROUNDS):
        train_A = ((round + 1) % 2 == 0)
        train_B = not train_A
        client_As, client_Bs = [], []
        server.broadcast_AB(clients)
        for client in clients:
            client.train_local(LOCAL_STEPS, LR, train_A=train_A, train_B=train_B)
            client_As.append(client.A)
            client_Bs.append(client.B)
        if train_B: server.aggregate_B(client_Bs)
        if train_A: server.aggregate_A(client_As)
        traj.append((server.A, server.B))
    trajectory_results['RoLoRA'] = traj

    # === 3. FedLoRA2 (No Rotation) ===
    server, clients = init_sim_state()
    traj = [(server.A, server.B)]
    for _ in range(NUM_ROUNDS):
        client_As, client_Bs = [], []
        server.broadcast_AB(clients)
        for client in clients:
            client.train_local(LOCAL_STEPS, LR, train_A=True, train_B=True)
            client_As.append(client.A)
            client_Bs.append(client.B)
        server.aggregate_AB(client_As, client_Bs)
        traj.append((server.A, server.B))
    trajectory_results['FedLoRA2 (No Rot)'] = traj

    # === 4. FedLoRA2 (With Rotation) ===
    server, clients = init_sim_state()
    traj = [(server.A, server.B)]
    A_ref, B_ref = server.A, server.B
    for round in range(NUM_ROUNDS):
        client_As, client_Bs = [], []
        server.broadcast_AB(clients)
        for client in clients:
            client.train_local(LOCAL_STEPS, LR, train_A=True, train_B=True)
            if (round + 1) % 2 != 0:
                a, b = client.align_A_and_get(A_ref)
            else:
                a, b = client.align_B_and_get(B_ref)
            client_As.append(a)
            client_Bs.append(b)
        server.aggregate_AB(client_As, client_Bs)
        A_ref, B_ref = server.A, server.B
        traj.append((server.A, server.B))
    trajectory_results['FedLoRA2 (Rot)'] = traj

    return trajectory_results, SHARED_INITIAL_A, SHARED_INITIAL_B


def plot_trajectory_with_contours(trajectory_results, start_A, start_B):
    """
    Plots trajectories over a contour map of the Global Loss function.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # 1. Determine Plot Bounds dynamically
    all_A = [start_A]
    all_B = [start_B]
    for traj in trajectory_results.values():
        all_A.extend([t[0] for t in traj])
        all_B.extend([t[1] for t in traj])

    margin = 0.5
    min_A, max_A = min(all_A) - margin, max(all_A) + margin
    min_B, max_B = min(all_B) - margin, max(all_B) + margin

    # 2. Generate Meshgrid for Contours
    # Create a grid of A and B values
    A_grid = np.linspace(min_A, max_A, 200)
    B_grid = np.linspace(min_B, max_B, 200)
    AA, BB = np.meshgrid(A_grid, B_grid)

    # 3. Calculate Global Loss at every point on the grid
    # Loss = (B*A - Global_Optimal)^2
    Z = (AA * BB - GLOBAL_OPTIMAL) ** 2

    # 4. Plot Contours
    # We use LogNorm because loss functions usually drop very quickly near the valley.
    # This makes the gradients visible.
    contour = ax.contourf(AA, BB, Z, levels=50, cmap='GnBu', norm=LogNorm(), alpha=0.6)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Global Loss $(B \cdot A - \Delta W_{opt})^2$', rotation=270, labelpad=20)

    # 5. Plot Optimal Hyperbola (The "Valley" floor)
    # B = Optimal / A
    A_line = np.linspace(min_A, max_A, 400)
    # Remove values near 0 to avoid asymptote connection lines
    A_line = A_line[np.abs(A_line) > 0.05]
    B_line = GLOBAL_OPTIMAL / A_line

    # Filter B_line to stay within plot bounds for cleaner legend
    mask = (B_line >= min_B) & (B_line <= max_B)
    ax.plot(A_line[mask], B_line[mask], 'r--', linewidth=2, label='Global Optimal Manifold')

    # 6. Plot Trajectories
    markers = ['o', 's', '^', 'D']
    for i, (label, traj) in enumerate(trajectory_results.items()):
        A_vals = [t[0] for t in traj]
        B_vals = [t[1] for t in traj]
        marker = markers[i % len(markers)]

        ax.plot(A_vals, B_vals, marker=marker, linestyle='-', linewidth=2,
                markersize=6, alpha=0.9, label=label)

        # Mark the end point specifically
        ax.scatter(A_vals[-1], B_vals[-1], s=100, edgecolors='white', zorder=10)

    # 7. Mark Start Point
    ax.scatter(start_A, start_B, c='black', marker='*', s=300, zorder=10, label='Start')

    # Formatting
    ax.set_xlabel("Parameter A", fontsize=14)
    ax.set_ylabel("Parameter B", fontsize=14)
    ax.set_title(f"Optimization Trajectory on Global Loss Landscape\n(Target $A \\cdot B = {GLOBAL_OPTIMAL:.2f}$)",
                 fontsize=16)
    ax.set_xlim(min_A, max_A)
    ax.set_ylim(min_B, max_B)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig("fedlora_trajectory_contour.png", dpi=300)
    print("Plot saved to fedlora_trajectory_contour.png")
    plt.show()


if __name__ == "__main__":
    # Run simulation
    results, start_A, start_B = run_simulation_logic()

    # Plot results
    plot_trajectory_with_contours(results, start_A, start_B)