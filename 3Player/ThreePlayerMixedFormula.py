import random
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider, TextBox


def simulate_mixed_game(
    p1_start_chance,
    p2_start_chance,
    p3_start_chance,
    matrix_p1,
    matrix_p2,
    matrix_p3,
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

        p1_confidence_B = 1 - p1_confidence_A

        # PGD probabilities
        p2_confidence_B = 1 - p2_confidence_A
        p3_confidence_B = 1 - p3_confidence_A

        plot_history.append((p1_confidence_A, p2_confidence_A, p3_confidence_A))

        # =========================================================
        # REWARD BLOCK (CORRECTED TO TRUE 3-PLAYER JOINT OUTCOMES)
        # =========================================================

        # Player 1 plays A (0): Payoff relies on joint choice of P2 and P3
        reward_1_A = (
            p2_confidence_A * p3_confidence_A * matrix_p1[0][0] +
            p2_confidence_A * p3_confidence_B * matrix_p1[0][1] +
            p2_confidence_B * p3_confidence_A * matrix_p1[0][1] +
            p2_confidence_B * p3_confidence_B * matrix_p1[0][1]
        )

        # Player 1 plays B (1)
        reward_1_B = (
            p2_confidence_A * p3_confidence_A * matrix_p1[1][0] +
            p2_confidence_A * p3_confidence_B * matrix_p1[1][0] +
            p2_confidence_B * p3_confidence_A * matrix_p1[1][0] +
            p2_confidence_B * p3_confidence_B * matrix_p1[1][1]
        )

        # Player 2 plays A (0): Payoff relies on joint choice of P1 and P3
        reward_2_A = (
            p1_confidence_A * p3_confidence_A * matrix_p2[0][0] +
            p1_confidence_A * p3_confidence_B * matrix_p2[0][1] +
            p1_confidence_B * p3_confidence_A * matrix_p2[0][1] +
            p1_confidence_B * p3_confidence_B * matrix_p2[0][1]
        )

        # Player 2 plays B (1)
        reward_2_B = (
            p1_confidence_A * p3_confidence_A * matrix_p2[1][0] +
            p1_confidence_A * p3_confidence_B * matrix_p2[1][0] +
            p1_confidence_B * p3_confidence_A * matrix_p2[1][0] +
            p1_confidence_B * p3_confidence_B * matrix_p2[1][1]
        )

        # Player 3 plays A (0): Payoff relies on joint choice of P1 and P2
        reward_3_A = (
            p1_confidence_A * p2_confidence_A * matrix_p3[0][0] +
            p1_confidence_A * p2_confidence_B * matrix_p3[0][1] +
            p1_confidence_B * p2_confidence_A * matrix_p3[0][1] +
            p1_confidence_B * p2_confidence_B * matrix_p3[0][1]
        )

        # Player 3 plays B (1)
        reward_3_B = (
            p1_confidence_A * p2_confidence_A * matrix_p3[1][0] +
            p1_confidence_A * p2_confidence_B * matrix_p3[1][0] +
            p1_confidence_B * p2_confidence_A * matrix_p3[1][0] +
            p1_confidence_B * p2_confidence_B * matrix_p3[1][1]
        )

        # Noise
        reward_1_A += random.uniform(-noise_level, noise_level)
        reward_1_B += random.uniform(-noise_level, noise_level)
        reward_2_A += random.uniform(-noise_level, noise_level)
        reward_2_B += random.uniform(-noise_level, noise_level)
        reward_3_A += random.uniform(-noise_level, noise_level)
        reward_3_B += random.uniform(-noise_level, noise_level)

        # 5. Perform Strategy Updates
        # Player 1 (MWU Score Update)
        p1_score_A += learning_speed * reward_1_A
        p1_score_B += learning_speed * reward_1_B

        p2_confidence_A += learning_speed * (reward_2_A - reward_2_B)
        p2_confidence_A = max(0, min(1, p2_confidence_A))

        p3_confidence_A += learning_speed * (reward_3_A - reward_3_B)
        p3_confidence_A = max(0, min(1, p3_confidence_A))

    return plot_history


def compute_theoretical_separatrix_plane(m1, m2, m3):
    """
    Directly computes the analytical plane coefficients Ax + By + Cz = D
    using the stable manifold left-eigenspace formulas.
    """
    # Defensive bound extraction to prevent division by zero or negative sqrt
    a1, b1 = max(1e-5, m1[0][0]), max(1e-5, m1[1][1])
    a2, b2 = max(1e-5, m2[0][0]), max(1e-5, m2[1][1])
    a3, b3 = max(1e-5, m3[0][0]), max(1e-5, m3[1][1])

    # Calculate exact asymmetric fixed point coordinates
    X = math.sqrt((a1 * b2 * b3) / (b1 * a2 * a3))
    Y = math.sqrt((b1 * a2 * b3) / (a1 * b2 * a3))
    Z = math.sqrt((b1 * b2 * a3) / (a1 * a2 * b3))

    x_eq = X / (1.0 + X)
    y_eq = Y / (1.0 + Y)
    z_eq = Z / (1.0 + Z)

    # Populate Jacobian matrix entries based on system derivatives
    M12 = x_eq * (1.0 - x_eq) * (a1 * z_eq + b1 * (1.0 - z_eq))
    M13 = x_eq * (1.0 - x_eq) * (a1 * y_eq + b1 * (1.0 - y_eq))
    M21 = a2 * z_eq + b2 * (1.0 - z_eq)
    M23 = a2 * x_eq + b2 * (1.0 - x_eq)
    M31 = a3 * y_eq + b3 * (1.0 - y_eq)
    M32 = a3 * x_eq + b3 * (1.0 - x_eq)

    J = np.array([
        [0.0, M12, M13],
        [M21, 0.0, M23],
        [M31, M32, 0.0]
    ])

    # Left eigenvectors of J are right eigenvectors of J transposed
    eigenvalues, eigenvectors = np.linalg.eig(J.T)

    # Find the single real positive (unstable) eigenvalue pushing away from the separatrix
    real_pos_indices = np.where((eigenvalues.real > 0) & (np.abs(eigenvalues.imag) < 1e-5))[0]

    if len(real_pos_indices) > 0:
        idx = real_pos_indices[0]
    else:
        idx = np.argmax(eigenvalues.real)  # Fallback to largest real component

    v = eigenvectors[:, idx].real
    A, B, C = v[0], v[1], v[2]

    # Standardize orientation direction
    if A < 0:
        A, B, C = -A, -B, -C

    D = A * x_eq + B * y_eq + C * z_eq

    print(f"Calculated Asymmetric Plane: {A:.4f}x + {B:.4f}y + {C:.4f}z = {D:.4f}")
    return A, B, C, D, x_eq, y_eq, z_eq


def calculate_plane_polygon(A, B, C, D):
    points = []
    for y in [0, 1]:
        for z in [0, 1]:
            if A != 0:
                x = (D - B * y - C * z) / A
                if 0 <= x <= 1: points.append([x, y, z])
    for x in [0, 1]:
        for z in [0, 1]:
            if B != 0:
                y = (D - A * x - C * z) / B
                if 0 <= y <= 1: points.append([x, y, z])
    for x in [0, 1]:
        for y in [0, 1]:
            if C != 0:
                z = (D - A * x - B * y) / C
                if 0 <= z <= 1: points.append([x, y, z])

    if not points: return []
    unique_points = np.unique(np.round(points, decimals=5), axis=0)
    if len(unique_points) < 3: return []

    centroid = np.mean(unique_points, axis=0)
    normal = np.array([A, B, C]) / np.linalg.norm([A, B, C])

    v1 = np.cross(normal, [1, 0, 0]) if abs(normal[0]) < 0.9 else np.cross(normal, [0, 1, 0])
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)

    angles = [np.arctan2(np.dot(p - centroid, v2), np.dot(p - centroid, v1)) for p in unique_points]
    return unique_points[np.argsort(angles)].tolist()


payoff_matrix_p1 = [[1, 0],
                    [0, 1]]
payoff_matrix_p2 = [[1, 0],
                    [0, 1]]
payoff_matrix_p3 = payoff_matrix_p1

# 1. Instantly calculate analytical coefficients
print("Evaluating system eigenvalues...")
A_th, B_th, C_th, D_th, x_eq, y_eq, z_eq = compute_theoretical_separatrix_plane(
    payoff_matrix_p1, payoff_matrix_p2, payoff_matrix_p3
)

# 2. Generate random starting conditions for visualization
random.seed(42)
test_scenarios = [[random.uniform(0.01, 0.99) for _ in range(3)] for _ in range(100)]

# 3. Plotting Setup
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

plt.subplots_adjust(bottom=0.28)
init_lr = 0.1
init_noise = 0.0

print("Running trajectory simulations...")
for p1, p2, p3 in test_scenarios:
    history = simulate_mixed_game(p1, p2, p3, payoff_matrix_p1, payoff_matrix_p2, payoff_matrix_p3)
    x = [h[0] for h in history]
    y = [h[1] for h in history]
    z = [h[2] for h in history]
    ax.plot(x, y, z, color="navy", linewidth=1, alpha=0.4)
    ax.scatter(p1, p2, p3, color='black', s=10, zorder=3)

# Real asymmetric interior fixed point
ax.scatter([x_eq], [y_eq], [z_eq], color='red', s=60, zorder=5, label='Saddle Equilibrium')

vertices_th = calculate_plane_polygon(A_th, B_th, C_th, D_th)
if vertices_th:
    separatrix_th = Poly3DCollection(
        [vertices_th], alpha=0.3,
        facecolors='crimson', edgecolors='red', linewidths=1.5
    )
    separatrix_th.set_label('Theoretical Separatrix Plane')
    ax.add_collection3d(separatrix_th)




def redraw_simulation(val=None):

    ax.clear()

    lr = slider_lr.val
    noise = slider_noise.val

    try:
        m1 = eval(text_p1.text)
        m2 = eval(text_p2.text)
        m3 = eval(text_p3.text)
    except Exception:
        m1 = payoff_matrix_p1
        m2 = payoff_matrix_p2
        m3 = payoff_matrix_p3

    # recompute separatrix
    np.random.seed(42)
    random.seed(42)

    A_sep, B_sep, C_sep, D_sep, x_eq_curr, y_eq_curr, z_eq_curr = compute_theoretical_separatrix_plane(m1, m2, m3)

    for p1, p2, p3 in test_scenarios:

        history = simulate_mixed_game(
            p1,
            p2,
            p3,
            m1,
            m2,
            m3,
            learning_rate=lr,
            noise_level=noise
        )

        xs = [h[0] for h in history]
        ys = [h[1] for h in history]
        zs = [h[2] for h in history]

        ax.plot(xs, ys, zs,
                color="navy",
                linewidth=1,
                alpha=0.6)

        ax.scatter(
            p1,
            p2,
            p3,
            color="black",
            s=15
        )

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

    ax.scatter(
        [x_eq_curr], [y_eq_curr], [z_eq_curr],
        color="red", s=50, label="Unstable Equilibrium"
    )

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)

    ax.set_xlabel("Player 1 confidence (MWU)")
    ax.set_ylabel("Player 2 confidence (PGD)")
    ax.set_zlabel("Player 3 confidence (PGD)")

    ax.set_title("Mixed Learning Dynamics")

    ax.view_init(elev=25, azim=-45)

    ax.legend()

    fig.canvas.draw_idle()

ax_lr = plt.axes([0.25,0.16,0.60,0.025])
ax_noise = plt.axes([0.25,0.12,0.60,0.025])

slider_lr = Slider(
    ax_lr,
    "Learning Rate",
    0.01,
    1.0,
    valinit=init_lr,
    valstep=0.01
)

slider_noise = Slider(
    ax_noise,
    "Noise",
    0.0,
    1.0,
    valinit=init_noise,
    valstep=0.01
)

ax_p1 = plt.axes([0.05,0.03,0.25,0.05])
ax_p2 = plt.axes([0.37,0.03,0.25,0.05])
ax_p3 = plt.axes([0.69,0.03,0.25,0.05])

text_p1 = TextBox(
    ax_p1,
    "P1",
    initial=str(payoff_matrix_p1)
)

text_p2 = TextBox(
    ax_p2,
    "P2",
    initial=str(payoff_matrix_p2)
)

text_p3 = TextBox(
    ax_p3,
    "P3",
    initial=str(payoff_matrix_p3)
)

slider_lr.on_changed(redraw_simulation)
slider_noise.on_changed(redraw_simulation)

text_p1.on_submit(redraw_simulation)
text_p2.on_submit(redraw_simulation)
text_p3.on_submit(redraw_simulation)

redraw_simulation()


plt.grid(True)
plt.show()