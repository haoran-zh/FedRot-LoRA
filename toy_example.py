import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
NUM_ROUNDS = 30  # Total communication rounds
LOCAL_STEPS = 30  # Local training steps per round
LR = 0.01  # Learning rate for local training
CLIENT_OPTIMALS = [0.5, 1.0, 1.5]  # Local optimal dW for each client
NUM_CLIENTS = len(CLIENT_OPTIMALS)
GLOBAL_OPTIMAL = np.mean(CLIENT_OPTIMALS)  # Global optimal is the average
EPSILON = 0  # For safe division



# Set a random seed for reproducibility
np.random.seed(5)


class Client:
    """
    Represents a single client in the FL simulation.

    Each client has its own local optimal delta_W and maintains
    its local LoRA parameters A and B.
    """

    def __init__(self, client_id, local_optimal_dw):
        self.client_id = client_id
        self.local_optimal_dw = local_optimal_dw

        # Initialize A with Gaussian noise and B with 0, as specified
        # Set to 0.0, as they will be overwritten by init_sim_state
        self.A = 0.0
        self.B = 0.0

    def set_AB(self, A, B):
        """Set local parameters from the server."""
        self.A = A
        self.B = B

    def train_local(self, num_steps, lr, train_A, train_B):
        """
        Perform local training steps.

        The loss function is L = (B*A - local_optimal_dw)^2
        Gradients:
        dL/dB = 2 * (B*A - local_optimal_dw) * A
        dL/dA = 2 * (B*A - local_optimal_dw) * B
        """
        for _ in range(num_steps):
            current_dw = self.B * self.A
            error = current_dw - self.local_optimal_dw

            # Store current values before update if both are trained
            A_old = self.A
            B_old = self.B

            if train_B:
                # Use A_old for gradient calculation if A is also being trained
                grad_B = 2 * error * (A_old if train_A else self.A)
                self.B -= lr * grad_B

            if train_A:
                # Use B_old for gradient calculation if B is also being trained
                grad_A = 2 * error * (B_old if train_B else self.B)
                self.A -= lr * grad_A


    def align_A_and_get(self, A_ref):
        """
        Perform scalar 'rotation' (alignment) on A, as described in Option 4.

        We find a scaling factor R such that self.A * R = A_ref.
        Then we return A' = self.A * R  (which is A_ref)
        and         B' = self.B / R

        This ensures A'*B' = (self.A * R) * (self.B / R) = self.A * self.B,
        preserving the product while aligning A.
        """
        # --- Rotation (Alignment) ---
        # Find R such that A_local * R = A_ref => R = A_ref / A_local
        # Avoid division by zero if A_local or A_ref is near zero
        if abs(self.A) < EPSILON or abs(A_ref) < EPSILON:
            # If alignment is unstable, just return unaligned values
            return self.A, self.B

        R = A_ref / self.A

        # --- De-rotation ---
        # A' = A_local * R
        # B' = B_local / R
        A_prime = self.A * R  # This will be equal to A_ref
        B_prime = self.B / (R + EPSILON)  # Add epsilon for safety

        return A_prime, B_prime

    def align_B_and_get(self, B_ref):
        """
        Perform scalar 'rotation' (alignment) on B.

        We find a scaling factor R such that self.B * R = B_ref.
        Then we return A' = self.A / R
        and         B' = self.B * R  (which is B_ref)

        This ensures A'*B' = (self.A / R) * (self.B * R) = self.A * self.B,
        preserving the product while aligning B.
        """
        # --- Rotation (Alignment) ---
        # Find R such that B_local * R = B_ref => R = B_ref / B_local
        # Avoid division by zero if B_local or B_ref is near zero
        if abs(self.B) < EPSILON or abs(B_ref) < EPSILON:
            # If alignment is unstable, just return unaligned values
            return self.A, self.B

        R = B_ref / self.B

        # --- De-rotation ---
        # B' = B_local * R
        # A' = A_local / R
        B_prime = self.B * R  # This will be equal to B_ref
        A_prime = self.A / (R + EPSILON)  # Add epsilon for safety

        return A_prime, B_prime


class Server:
    """
    Represents the central server.

    It initializes global parameters, aggregates client updates,
    and broadcasts the updated global model.
    """

    def __init__(self, num_clients):
        self.num_clients = num_clients
        # Global A and B, initialized same as clients
        # Set to 0.0, as they will be overwritten by init_sim_state
        self.A = 0.0
        self.B = 0.0

    def aggregate_B(self, client_Bs):
        """Aggregate only B using FedAvg."""
        self.B = np.mean(client_Bs)

    def aggregate_A(self, client_As):
        """Aggregate only A using FedAvg."""
        self.A = np.mean(client_As)

    def aggregate_AB(self, client_As, client_Bs):
        """
        Aggregate both A and B using FedAvg.
        This interprets "delta W=(sum B)(sum A)" as
        B_global = mean(B_i) and A_global = mean(A_i),
        so dW_global = B_global * A_global.
        """
        self.A = np.mean(client_As)
        self.B = np.mean(client_Bs)

    def broadcast_AB(self, clients):
        """Send the current global A and B to all clients."""
        for client in clients:
            client.set_AB(self.A, self.B)

    def get_global_dw(self):
        """Return the effective global delta W."""
        return self.B * self.A


def run_simulation():
    """
    Runs the full simulation for all four options and plots the results.
    """

    # Store results for plotting
    results = {}
    trajectory_results = {}  # NEW: Store (A, B) tuples for trajectory plot

    # --- Define a single set of initial parameters for all methods ---
    # We use the now-deterministic RNG (seeded at top level) to get one initial A.
    SHARED_INITIAL_A = np.random.randn()
    SHARED_INITIAL_B = 0.0

    # --- Helper function to reset simulation state ---
    def init_sim_state():
        server = Server(NUM_CLIENTS)
        # Explicitly set to the *same* shared initial values
        server.A = SHARED_INITIAL_A
        server.B = SHARED_INITIAL_B

        clients = [Client(i, CLIENT_OPTIMALS[i]) for i in range(NUM_CLIENTS)]

        # Synchronize all clients with the shared initial A and B
        for client in clients:
            client.set_AB(SHARED_INITIAL_A, SHARED_INITIAL_B)

        return server, clients

    # === Option 1: FFA-LoRA (Freeze A, train B) ===
    print("--- Running Option 1: FFA-LoRA ---")
    server, clients = init_sim_state()
    history_ffa = []
    traj_ffa = [(server.A, server.B)]  # NEW: Store initial (A, B)

    # Note: Global A is frozen at its initial value for all rounds
    frozen_A = server.A
    print(f"Initial (Frozen) A = {frozen_A:.4f}, Initial B = {server.B:.4f}")

    for round in range(NUM_ROUNDS):
        client_Bs = []
        for client in clients:
            # Broadcast global B. A is frozen, so we use the initial frozen_A
            client.set_AB(frozen_A, server.B)

            # Local training: Freeze A, train B
            client.train_local(num_steps=LOCAL_STEPS, lr=LR, train_A=False, train_B=True)
            client_Bs.append(client.B)

        # Server aggregates B. A remains unchanged.
        server.aggregate_B(client_Bs)

        global_dw = server.B * frozen_A
        history_ffa.append(global_dw)
        traj_ffa.append((frozen_A, server.B))  # NEW: Store (A, B)
        if (round + 1) % 5 == 0:
            print(f"Round {round + 1}: Global dW = {global_dw:.4f} (A={frozen_A:.4f}, B={server.B:.4f})")
    results['FFA-LoRA (Freeze A)'] = history_ffa
    trajectory_results['FFA-LoRA (Freeze A)'] = traj_ffa  # NEW

    # === Option 2: RoLoRA (Alternate training A and B) ===
    print("\n--- Running Option 2: RoLoRA ---")
    server, clients = init_sim_state()
    history_rolora = []
    traj_rolora = [(server.A, server.B)]  # NEW: Store initial (A, B)
    print(f"Initial A = {server.A:.4f}, Initial B = {server.B:.4f}")

    for round in range(NUM_ROUNDS):
        # Odd rounds (1, 3, ...): Freeze A, Train B
        if (round + 1) % 2 != 0:
            train_A_flag, train_B_flag = False, True
            train_desc = "Train B"
        # Even rounds (2, 4, ...): Freeze B, Train A
        else:
            train_A_flag, train_B_flag = True, False
            train_desc = "Train A"

        client_As = []
        client_Bs = []

        # Broadcast current global model
        server.broadcast_AB(clients)

        for client in clients:
            client.train_local(num_steps=LOCAL_STEPS, lr=LR, train_A=train_A_flag, train_B=train_B_flag)
            client_As.append(client.A)
            client_Bs.append(client.B)

        # Server aggregates only the parameter that was trained
        if train_B_flag:
            server.aggregate_B(client_Bs)
        if train_A_flag:
            server.aggregate_A(client_As)

        global_dw = server.get_global_dw()
        history_rolora.append(global_dw)
        traj_rolora.append((server.A, server.B))  # NEW: Store (A, B)
        if (round + 1) % 5 == 0:
            print(f"Round {round + 1} ({train_desc}): Global dW = {global_dw:.4f} (A={server.A:.4f}, B={server.B:.4f})")
    results['RoLoRA (Alternate)'] = history_rolora
    trajectory_results['RoLoRA (Alternate)'] = traj_rolora  # NEW

    # === Option 3: FedLoRA2 without rotation ===
    print("\n--- Running Option 3: FedLoRA2 (no rotation) ---")
    server, clients = init_sim_state()
    history_fedlora = []
    traj_fedlora = [(server.A, server.B)]  # NEW: Store initial (A, B)
    print(f"Initial A = {server.A:.4f}, Initial B = {server.B:.4f}")

    for round in range(NUM_ROUNDS):
        client_As = []
        client_Bs = []

        # Broadcast current global model
        server.broadcast_AB(clients)

        for client in clients:
            # Train both A and B
            client.train_local(num_steps=LOCAL_STEPS, lr=LR, train_A=True, train_B=True)
            client_As.append(client.A)
            client_Bs.append(client.B)

        # Server aggregates both A and B
        server.aggregate_AB(client_As, client_Bs)

        global_dw = server.get_global_dw()
        history_fedlora.append(global_dw)
        traj_fedlora.append((server.A, server.B))  # NEW: Store (A, B)
        if (round + 1) % 5 == 0:
            print(f"Round {round + 1}: Global dW = {global_dw:.4f} (A={server.A:.4f}, B={server.B:.4f})")
    results['FedLoRA2 (No Rotation)'] = history_fedlora
    trajectory_results['FedLoRA2 (No Rotation)'] = traj_fedlora  # NEW

    # === Option 4: FedLoRA2 with rotation (Alternating) ===
    print("\n--- Running Option 4: FedLoRA2 (with alternating rotation) ---")
    server, clients = init_sim_state()
    history_fedlora_rot = []
    traj_fedlora_rot = [(server.A, server.B)]  # NEW: Store initial (A, B)

    # Reference model for alignment is the global model from the *previous* round
    A_ref = server.A
    B_ref = server.B
    print(f"Initial A_ref = {A_ref:.4f}, Initial B_ref = {B_ref:.4f}")

    for round in range(NUM_ROUNDS):
        client_As_aligned = []
        client_Bs_aligned = []

        # Broadcast current global model
        server.broadcast_AB(clients)

        for client in clients:
            # Train both A and B
            client.train_local(num_steps=LOCAL_STEPS, lr=LR, train_A=True, train_B=True)

            # Align after local training, before sending to server
            # if round == 0:
            #     # First round, no rotation, as specified
            #     A_prime, B_prime = client.A, client.B
            if (round + 1) % 2 != 0:
                # Odd rounds (1, 3, ...): Align A
                A_prime, B_prime = client.align_A_and_get(A_ref)
            else:
                # Even rounds (2, 4, ...): Align B
                A_prime, B_prime = client.align_B_and_get(B_ref)

            client_As_aligned.append(A_prime)
            client_Bs_aligned.append(B_prime)

        # Server aggregates the *aligned* parameters
        server.aggregate_AB(client_As_aligned, client_Bs_aligned)

        # Update the reference A and B for the *next* round's alignment
        A_ref = server.A
        B_ref = server.B

        global_dw = server.get_global_dw()
        history_fedlora_rot.append(global_dw)
        traj_fedlora_rot.append((server.A, server.B))  # NEW: Store (A, B)
        print(f"Round {round + 1}: Global dW = {global_dw:.4f} (A={server.A:.4f}, B={server.B:.4f})")
    results['FedLoRA2 (With Alt. Rotation)'] = history_fedlora_rot
    trajectory_results['FedLoRA2 (With Alt. Rotation)'] = traj_fedlora_rot  # NEW

    # --- Plotting ---
    # PLOT 1: dW vs. Rounds

    # Plot the target optimal value
    plt.axhline(y=GLOBAL_OPTIMAL, color='r', linestyle='--',
                label=f'Global Optimal dW ({GLOBAL_OPTIMAL:.2f})')

    plt.xlabel("Communication Round")
    plt.ylabel("Global $\Delta W = B \cdot A$")
    plt.title("Comparison of Federated LoRA Strategies (Scalar Toy Example)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(range(0, NUM_ROUNDS + 1, 2))
    plt.xlim(0, NUM_ROUNDS + 1)
    plt.tight_layout()

    # Save the plot
    plot_filename = "fedlora_toy_example_comparison.png"
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")

    # --- NEW PLOT: (A, B) Trajectory ---
    plt.figure(figsize=(12, 10))
    for label, trajectory in trajectory_results.items():
        A_vals = [t[0] for t in trajectory]
        B_vals = [t[1] for t in trajectory]
        plt.plot(A_vals, B_vals, '.-', label=label, markersize=8, alpha=0.7)

    # Plot the start point
    plt.plot(SHARED_INITIAL_A, SHARED_INITIAL_B, 'k*', markersize=15,
             label=f'Start Point ({SHARED_INITIAL_A:.2f}, {SHARED_INITIAL_B:.2f})')

    # Plot the optimal manifold (hyperbola B = GLOBAL_OPTIMAL / A)
    # Get current plot limits to draw the hyperbola
    ax_limits = plt.gca().get_xlim()
    A_plot_range = np.linspace(ax_limits[0], ax_limits[1], 200)
    # Avoid division by zero
    A_plot_range = A_plot_range[np.abs(A_plot_range) > 0.01]
    B_optimal_line = GLOBAL_OPTIMAL / (A_plot_range + EPSILON)

    plt.plot(A_plot_range, B_optimal_line, 'r--',
             label=f'Global Optimal Manifold (B*A = {GLOBAL_OPTIMAL:.2f})')

    # Set plot limits again to ensure hyperbola doesn't dominate
    y_ax_limits = plt.gca().get_ylim()
    # Clip y-axis to avoid extreme values from hyperbola
    plt.ylim(max(-20, y_ax_limits[0]), min(20, y_ax_limits[1]))
    plt.xlim(ax_limits)

    plt.xlabel("Parameter A", fontsize=18)
    plt.ylabel("Parameter B", fontsize=18)
    plt.title("Global (A, B) Parameter Trajectory", fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.tight_layout()
    # Save the trajectory plot
    plot_filename_traj = "fedlora_toy_example_trajectory.png"
    plt.savefig(plot_filename_traj)
    print(f"Trajectory plot saved to {plot_filename_traj}")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    run_simulation()