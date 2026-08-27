import random
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider, TextBox


def get_expected_payoffs(p_other1, p_other2, matrix):
    """Calculates expected payoffs for choosing A or B given the other two players' A-probabilities."""
    q_other1 = 1.0 - p_other1
    q_other2 = 1.0 - p_other2

    # Expected payoff for playing Action A
    prob_both_A = p_other1 * p_other2
    payoff_A = prob_both_A * matrix[0][0] + (1.0 - prob_both_A) * matrix[0][1]

    # Expected payoff for playing Action B
    prob_both_B = q_other1 * q_other2
    payoff_B = (1.0 - prob_both_B) * matrix[1][0] + prob_both_B * matrix[1][1]

    return payoff_A, payoff_B


def simulate_mixed_game(
    p1_start_chance,
    p2_start_chance,
    p3_start_chance,
    matrix,
    learning_rate=0.1,
    iterations=1000,
    noise_level = 0.1
):
    # --- PLAYER 1 (MWU) INITIALIZATION ---
    p1_start_chance = max(0.001, min(0.999, p1_start_chance))
    p1_score_A = math.log(p1_start_chance)
    p1_score_B = math.log(1 - p1_start_chance)

    # --- PLAYER 2 & 3 (PGD) INITIALIZATION ---
    p2_confidence_A = p2_start_chance
    p3_confidence_A = p3_start_chance

    plot_history = []

    for i in range(iterations):
        learning_speed = learning_rate / math.sqrt(i + 1)

        # MWU probabilities (Player 1)
        try:
            denom1 = math.exp(p1_score_A) + math.exp(p1_score_B)
            p1_confidence_A = math.exp(p1_score_A) / denom1
        except OverflowError:
            max_score = max(p1_score_A, p1_score_B)
            denom1 = math.exp(p1_score_A - max_score) + math.exp(p1_score_B - max_score)
            p1_confidence_A = math.exp(p1_score_A - max_score) / denom1

        plot_history.append((p1_confidence_A, p2_confidence_A, p3_confidence_A))

        # Expected Payoffs using simplified helper
        reward_1_A, reward_1_B = get_expected_payoffs(p2_confidence_A, p3_confidence_A, matrix)
        reward_2_A, reward_2_B = get_expected_payoffs(p1_confidence_A, p3_confidence_A, matrix)
        reward_3_A, reward_3_B = get_expected_payoffs(p1_confidence_A, p2_confidence_A, matrix)

        # Add Noise
        if noise_level > 0:
            reward_1_A += random.uniform(-noise_level, noise_level)
            reward_1_B += random.uniform(-noise_level, noise_level)
            reward_2_A += random.uniform(-noise_level, noise_level)
            reward_2_B += random.uniform(-noise_level, noise_level)
            reward_3_A += random.uniform(-noise_level, noise_level)
            reward_3_B += random.uniform(-noise_level, noise_level)

        # Perform Strategy Updates
        p1_score_A += learning_speed * reward_1_A
        p1_score_B += learning_speed * reward_1_B

        p2_confidence_A += learning_speed * (reward_2_A - reward_2_B)
        p2_confidence_A = max(0, min(1, p2_confidence_A))

        p3_confidence_A += learning_speed * (reward_3_A - reward_3_B)
        p3_confidence_A = max(0, min(1, p3_confidence_A))

        if (
                abs(p1_confidence_A - round(p1_confidence_A)) < 1e-4 and
                abs(p2_confidence_A - round(p2_confidence_A)) < 1e-4 and
                abs(p3_confidence_A - round(p3_confidence_A)) < 1e-4
        ):
            break

    return plot_history


def detect_attractor(final_state, threshold=0.5):
    """Returns True if ALL individual players converged past the threshold to the A-attractor."""
    return all(p > threshold for p in final_state)


def compute_separatrix_plane(matrix, n_probes=80, bisection_steps=8,
                              learning_rate=0.1, iters=400):
    """Empirically locates the separatrix via bisection search and SVD plane fitting."""
    def final_state(p1, p2, p3):
        hist = simulate_mixed_game(p1, p2, p3, matrix,
                                   learning_rate=learning_rate, iterations=iters,
                                   noise_level=0.0)
        return hist[-1]

    def in_basin_A(p):
        return detect_attractor(final_state(*p))

    boundary_pts = []
    attempts = 0
    while len(boundary_pts) < n_probes and attempts < n_probes * 10:
        attempts += 1
        a = np.random.uniform(0.05, 0.95, 3)
        b = np.random.uniform(0.05, 0.95, 3)
        ba = in_basin_A(a)
        bb = in_basin_A(b)
        if ba == bb:
            continue

        # Bisect to find the precise boundary crossing
        lo, hi = a.copy(), b.copy()
        lo_basin = ba
        for _ in range(bisection_steps):
            mid = (lo + hi) / 2.0
            if in_basin_A(mid) == lo_basin:
                lo = mid
            else:
                hi = mid
        boundary_pts.append((lo + hi) / 2.0)

    if len(boundary_pts) < 3:
        raise ValueError(f"Could only find {len(boundary_pts)} boundary points.")

    pts = np.array(boundary_pts)
    centroid = pts.mean(axis=0)

    # SVD to find the plane normal
    _, _, Vt = np.linalg.svd(pts - centroid)
    normal = Vt[-1]
    D = float(np.dot(normal, centroid))
    A, B, C = float(normal[0]), float(normal[1]), float(normal[2])

    residuals = np.abs(pts @ normal - D)
    print(f"  Separatrix plane fit: {A:.4f}x + {B:.4f}y + {C:.4f}z = {D:.4f}")
    print(f"  Points used: {len(pts)} | Residuals: mean={residuals.mean():.4f}, max={residuals.max():.4f}")

    return A, B, C, D, pts


def calculate_plane_polygon(A, B, C, D):
    """Finds intersection vertices of the plane Ax + By + Cz = D with the unit cube [0,1]^3."""
    points = []

    for y in [0, 1]:
        for z in [0, 1]:
            if A != 0:
                x = (D - B * y - C * z) / A
                if 0 <= x <= 1:
                    points.append([x, y, z])

    for x in [0, 1]:
        for z in [0, 1]:
            if B != 0:
                y = (D - A * x - C * z) / B
                if 0 <= y <= 1:
                    points.append([x, y, z])

    for x in [0, 1]:
        for y in [0, 1]:
            if C != 0:
                z = (D - A * x - B * y) / C
                if 0 <= z <= 1:
                    points.append([x, y, z])

    if not points:
        return []

    unique_points = np.unique(np.round(points, decimals=5), axis=0)
    if len(unique_points) < 3:
        return []

    centroid = np.mean(unique_points, axis=0)
    normal = np.array([A, B, C]) / np.linalg.norm([A, B, C])

    if abs(normal[0]) < 0.9:
        v1 = np.cross(normal, [1, 0, 0])
    else:
        v1 = np.cross(normal, [0, 1, 0])
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)

    angles = [np.arctan2(np.dot(p - centroid, v2), np.dot(p - centroid, v1))
              for p in unique_points]
    return unique_points[np.argsort(angles)].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SETUP
# ─────────────────────────────────────────────────────────────────────────────

payoff_matrix = [[1, 0],
                 [0, 1]]

print("Computing data-driven separatrix (this may take a moment)...")
np.random.seed(42)
random.seed(42)

A_sep, B_sep, C_sep, D_sep, sep_pts = compute_separatrix_plane(
    payoff_matrix,
    n_probes=250,
    bisection_steps=15,
    iters=2000,
)

test_scenarios = []
for _ in range(500):
    p1 = random.uniform(0.01, 0.99)
    p2 = random.uniform(0.01, 0.99)
    p3 = random.uniform(0.01, 0.99)
    test_scenarios.append((p1, p2, p3))

# Plot setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.28)

init_lr = 0.1
init_noise = 0.0


def redraw_simulation(val=None):
    ax.clear()
    lr = slider_lr.val
    noise = slider_noise.val

    try:
        m = eval(text_matrix.text)
    except Exception:
        m = payoff_matrix

    np.random.seed(42)
    random.seed(42)

    # Recompute separatrix based on current matrix
    A_sep, B_sep, C_sep, D_sep, _ = compute_separatrix_plane(
        m,
        n_probes=250,
        bisection_steps=15,
        learning_rate=lr,
        iters=2000
    )

    # Plot trajectories
    for p1, p2, p3 in test_scenarios:
        history = simulate_mixed_game(p1, p2, p3, m, learning_rate=lr, noise_level=noise)
        xs = [h[0] for h in history]
        ys = [h[1] for h in history]
        zs = [h[2] for h in history]
        ax.plot(xs, ys, zs, color="navy", linewidth=1, alpha=0.6)
        ax.scatter(p1, p2, p3, color="black", s=15)

    # Add separatrix plane
    vertices = calculate_plane_polygon(A_sep, B_sep, C_sep, D_sep)
    if vertices:
        plane = Poly3DCollection(
            [vertices],
            alpha=0.35,
            facecolors="mediumseagreen",
            edgecolors="darkgreen",
            linewidths=1.5
        )
        plane.set_label("Separatrix")
        ax.add_collection3d(plane)

    # Unstable Equilibrium dot
    ax.scatter([0.5], [0.5], [0.5], color="red", s=50, label="Unstable Equilibrium", zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("Player 1 confidence (MWU)")
    ax.set_ylabel("Player 2 confidence (PGD)")
    ax.set_zlabel("Player 3 confidence (PGD)")
    ax.set_title("Mixed Learning Dynamics")
    ax.view_init(elev=25, azim=-45)
    ax.legend()

    # Count convergence destinations
    converged_A = 0
    converged_B = 0

    for p1, p2, p3 in test_scenarios:

        history = simulate_mixed_game(
            p1,
            p2,
            p3,
            m,
            learning_rate=lr,
            iterations=1000,
            noise_level=0.0  # deterministic basin measurement
        )

        final_p1, final_p2, final_p3 = history[-1]

        if final_p1 > 0.5 and final_p2 > 0.5 and final_p3 > 0.5:
            converged_A += 1
        elif final_p1 < 0.5 and final_p2 < 0.5 and final_p3 < 0.5:
            converged_B += 1

    total = len(test_scenarios)

    percentage_A = 100 * converged_A / total
    percentage_B = 100 * converged_B / total

    print(f"Trajectories converging to A (1,1,1): {converged_A}/{total} ({percentage_A:.2f}%)")
    print(f"Trajectories converging to B (0,0,0): {converged_B}/{total} ({percentage_B:.2f}%)")


    fig.canvas.draw_idle()


# Interactive UI controls
ax_lr = plt.axes([0.25, 0.16, 0.60, 0.025])
ax_noise = plt.axes([0.25, 0.12, 0.60, 0.025])

slider_lr = Slider(ax_lr, "Learning Rate", 0.01, 1.0, valinit=init_lr, valstep=0.01)
slider_noise = Slider(ax_noise, "Noise", 0.0, 1.0, valinit=init_noise, valstep=0.01)

# Combined, single text box for symmetric payoffs
ax_matrix = plt.axes([0.25, 0.03, 0.60, 0.05])
text_matrix = TextBox(ax_matrix, "Shared Payoff", initial=str(payoff_matrix))

slider_lr.on_changed(redraw_simulation)
slider_noise.on_changed(redraw_simulation)
text_matrix.on_submit(redraw_simulation)

# Run initial rendering
redraw_simulation()
plt.grid(True)
plt.show()
print("Done. Plot saved.")