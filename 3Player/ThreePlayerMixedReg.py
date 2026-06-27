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
        # REWARD BLOCK (NOW MATRIX-BASED, NO HARD-CODING)
        # =========================================================

        reward_1_A = (
            p2_confidence_A * matrix_p1[0][0] +
            p2_confidence_B * matrix_p1[0][1] +
            p3_confidence_A * matrix_p1[0][0] +
            p3_confidence_B * matrix_p1[0][1]
        )

        reward_1_B = (
            p2_confidence_A * matrix_p1[1][0] +
            p2_confidence_B * matrix_p1[1][1] +
            p3_confidence_A * matrix_p1[1][0] +
            p3_confidence_B * matrix_p1[1][1]
        )

        reward_2_A = (
            p1_confidence_A * matrix_p2[0][0] +
            p1_confidence_B * matrix_p2[0][1] +
            p3_confidence_A * matrix_p2[0][0] +
            p3_confidence_B * matrix_p2[0][1]
        )

        reward_2_B = (
            p1_confidence_A * matrix_p2[1][0] +
            p1_confidence_B * matrix_p2[1][1] +
            p3_confidence_A * matrix_p2[1][0] +
            p3_confidence_B * matrix_p2[1][1]
        )

        reward_3_A = (
            p1_confidence_A * matrix_p3[0][0] +
            p1_confidence_B * matrix_p3[0][1] +
            p2_confidence_A * matrix_p3[0][0] +
            p2_confidence_B * matrix_p3[0][1]
        )

        reward_3_B = (
            p1_confidence_A * matrix_p3[1][0] +
            p1_confidence_B * matrix_p3[1][1] +
            p2_confidence_A * matrix_p3[1][0] +
            p2_confidence_B * matrix_p3[1][1]
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


def detect_attractor(final_state, threshold=0.5):
    """Returns True if the trajectory converged to the A-attractor (1,1,1)."""
    return sum(final_state) > 3 * threshold


def compute_separatrix_plane(matrix_p1, matrix_p2, matrix_p3,
                              n_probes=300, bisection_steps=15,
                              learning_rate=0.1, iters=2000):
    """
    Empirically locates the separatrix by:
      1. Sampling random pairs of starting conditions in opposite basins.
      2. Binary-searching (bisecting) along the line between each pair to find
         the exact basin boundary point.
      3. Fitting a best-fit plane through all boundary points via SVD.

    Returns (A, B, C, D) coefficients of the plane Ax + By + Cz = D,
    and the array of boundary points used for fitting.
    """
    def final_state(p1, p2, p3):
        hist = simulate_mixed_game(p1, p2, p3, matrix_p1, matrix_p2, matrix_p3,
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
            continue  # Same basin — no boundary crossing on this segment

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
        raise ValueError(
            f"Could only find {len(boundary_pts)} boundary points — "
            "not enough to fit a plane. Try different payoff matrices."
        )

    pts = np.array(boundary_pts)
    centroid = pts.mean(axis=0)

    # SVD: the right singular vector corresponding to the smallest singular value
    # is the normal to the best-fit plane through the point cloud.
    _, _, Vt = np.linalg.svd(pts - centroid)
    normal = Vt[-1]
    D = float(np.dot(normal, centroid))
    A, B, C = float(normal[0]), float(normal[1]), float(normal[2])

    residuals = np.abs(pts @ normal - D)
    print(f"  Separatrix plane fit: {A:.4f}x + {B:.4f}y + {C:.4f}z = {D:.4f}")
    print(f"  Points used: {len(pts)} | Residuals: mean={residuals.mean():.4f}, max={residuals.max():.4f}")

    return A, B, C, D, pts


def calculate_plane_polygon(A, B, C, D):
    """
    Finds intersection vertices of the plane Ax + By + Cz = D
    with the unit cube [0,1]^3, sorted angularly to form a polygon.
    """
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
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

payoff_matrix_p1 = [[1, 0],
                    [0, 1]]
payoff_matrix_p2 = payoff_matrix_p1
payoff_matrix_p3 = payoff_matrix_p1

# ── 1. Compute the separatrix from data ──────────────────────────────────────
print("Computing data-driven separatrix (this may take a moment)...")
np.random.seed(42)
random.seed(42)

A_sep, B_sep, C_sep, D_sep, sep_pts = compute_separatrix_plane(
    payoff_matrix_p1, payoff_matrix_p2, payoff_matrix_p3,
    n_probes=250,
    bisection_steps=15,
    iters=2000,
)

# ── 2. Run trajectory simulations ────────────────────────────────────────────
test_scenarios = []
for _ in range(100):
    p1 = random.uniform(0.01, 0.99)
    p2 = random.uniform(0.01, 0.99)
    p3 = random.uniform(0.01, 0.99)
    test_scenarios.append((p1, p2, p3))

# ── 3. Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

plt.subplots_adjust(bottom=0.28)
init_lr = 0.1
init_noise = 0.0

for p1, p2, p3 in test_scenarios:
    history = simulate_mixed_game(p1, p2, p3, payoff_matrix_p1, payoff_matrix_p2, payoff_matrix_p3)
    x = [h[0] for h in history]
    y = [h[1] for h in history]
    z = [h[2] for h in history]
    ax.plot(x, y, z, color="navy", linewidth=1, alpha=0.6)
    ax.scatter(p1, p2, p3, color='black', s=15, zorder=3)

# Unstable equilibrium
ax.scatter([0.5], [0.5], [0.5], color='red', s=50, zorder=5, label='Unstable Equilibrium')

# Data-driven separatrix plane
vertices = calculate_plane_polygon(A_sep, B_sep, C_sep, D_sep)
if vertices:
    separatrix = Poly3DCollection(
        [vertices], alpha=0.35,
        facecolors='mediumseagreen',
        edgecolors='darkgreen',
        linewidths=1.5
    )
    separatrix.set_label('Data-Driven Separatrix')
    ax.add_collection3d(separatrix)




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

    A_sep, B_sep, C_sep, D_sep, _ = compute_separatrix_plane(
        m1,
        m2,
        m3,
        n_probes=250,
        bisection_steps=15,
        learning_rate=lr,
        iters=2000
    )

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
        [0.5],
        [0.5],
        [0.5],
        color="red",
        s=50,
        label="Unstable Equilibrium"
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

ax.set_xlabel("Player 1 confidence (MWU)")
ax.set_ylabel("Player 2 confidence (PGD)")
ax.set_zlabel("Player 3 confidence (PGD)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.view_init(elev=25, azim=-45)
ax.legend()
plt.grid(True)
plt.show()
print("Done. Plot saved.")