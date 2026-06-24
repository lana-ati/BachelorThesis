import random
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def simulate_mixed_game(
        p1_start_chance,
        p2_start_chance,
        p3_start_chance,
        matrix_p1,
        matrix_p2,
        matrix_p3,
        learning_rate=0.1,
        iterations=1000,
        noise_level=0.1
):
    # Player 1 (MW)
    p1_start_chance = max(0.001, min(0.999, p1_start_chance))
    p1_score_A = math.log(p1_start_chance)
    p1_score_B = math.log(1 - p1_start_chance)

    # Player 2 and 3 (PGD)
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


        reward_1_A = p2_confidence_A * matrix_p1[0][0] + p2_confidence_B * matrix_p1[0][1] + p3_confidence_A * \
                     matrix_p1[0][0] + p3_confidence_B * matrix_p1[0][1]
        reward_1_B = p2_confidence_A * matrix_p1[1][0] + p2_confidence_B * matrix_p1[1][1] + p3_confidence_A * \
                     matrix_p1[1][0] + p3_confidence_B * matrix_p1[1][1]

        reward_2_A = p1_confidence_A * matrix_p2[0][0] + p1_confidence_B * matrix_p2[0][1] + p3_confidence_A * \
                     matrix_p2[0][0] + p3_confidence_B * matrix_p2[0][1]
        reward_2_B = p1_confidence_A * matrix_p2[1][0] + p1_confidence_B * matrix_p2[1][1] + p3_confidence_A * \
                     matrix_p2[1][0] + p3_confidence_B * matrix_p2[1][1]

        reward_3_A = p1_confidence_A * matrix_p3[0][0] + p1_confidence_B * matrix_p3[0][1] + p2_confidence_A * \
                     matrix_p3[0][0] + p2_confidence_B * matrix_p3[0][1]
        reward_3_B = p1_confidence_A * matrix_p3[1][0] + p1_confidence_B * matrix_p3[1][1] + p2_confidence_A * \
                     matrix_p3[1][0] + p2_confidence_B * matrix_p3[1][1]

        # Noise
        reward_1_A += random.uniform(-noise_level, noise_level)
        reward_1_B += random.uniform(-noise_level, noise_level)
        reward_2_A += random.uniform(-noise_level, noise_level)
        reward_2_B += random.uniform(-noise_level, noise_level)
        reward_3_A += random.uniform(-noise_level, noise_level)
        reward_3_B += random.uniform(-noise_level, noise_level)

        # Strategy Updates
        p1_score_A += learning_speed * reward_1_A
        p1_score_B += learning_speed * reward_1_B

        p2_confidence_A += learning_speed * (reward_2_A - reward_2_B)
        p2_confidence_A = max(0, min(1, p2_confidence_A))

        p3_confidence_A += learning_speed * (reward_3_A - reward_3_B)
        p3_confidence_A = max(0, min(1, p3_confidence_A))

    return plot_history


def compute_theoretical_separatrix_plane(matrix_p1):
    """
    Directly computes the analytical plane coefficients Ax + By + Cz = D
    using the stable manifold left-eigenspace formulas.
    """
    a = matrix_p1[0][0]
    b = matrix_p1[1][1]

    S = a + b
    # Dominant unstable eigenvalue from characteristic polynomial
    lambda_2 = (S + math.sqrt(S ** 2 + 8 * a * b)) / 2.0

    # Left eigenvector components
    A = 2 * S
    B = lambda_2
    C = lambda_2

    # Pivot plane around the interior mixed equilibrium saddle point (p, p, p)
    p = b / S
    D = A * p + B * p + C * p

    print(f"Calculated Theoretical Plane: {A:.4f}x + {B:.4f}y + {C:.4f}z = {D:.4f}")
    return A, B, C, D


def calculate_plane_polygon(A, B, C, D):
    """
    Finds intersection vertices of the plane Ax + By + Cz = D
    with the unit cube bounds [0,1]^3.
    """
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
A_th, B_th, C_th, D_th = compute_theoretical_separatrix_plane(payoff_matrix_p1)

# 2. Generate random starting conditions for visualization
random.seed(42)
test_scenarios = [[random.uniform(0.01, 0.99) for _ in range(3)] for _ in range(100)]

# 3. Plotting Setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

print("Running trajectory simulations...")
for p1, p2, p3 in test_scenarios:
    history = simulate_mixed_game(p1, p2, p3, payoff_matrix_p1, payoff_matrix_p2, payoff_matrix_p3)
    x = [h[0] for h in history]
    y = [h[1] for h in history]
    z = [h[2] for h in history]
    ax.plot(x, y, z, color="navy", linewidth=1, alpha=0.4)
    ax.scatter(p1, p2, p3, color='black', s=10, zorder=3)

# Internal fixed point
p_eq = payoff_matrix_p1[1][1] / (payoff_matrix_p1[0][0] + payoff_matrix_p1[1][1])
ax.scatter([p_eq], [p_eq], [p_eq], color='red', s=60, zorder=5, label='Saddle Equilibrium')

# Generate and render the theoretical plane slice
vertices_th = calculate_plane_polygon(A_th, B_th, C_th, D_th)
if vertices_th:
    separatrix_th = Poly3DCollection(
        [vertices_th], alpha=0.3,
        facecolors='crimson', edgecolors='red', linewidths=1.5
    )
    separatrix_th.set_label('Theoretical Separatrix Plane')
    ax.add_collection3d(separatrix_th)

ax.set_xlabel("Player 1 confidence (MWU)")
ax.set_ylabel("Player 2 confidence (PGD)")
ax.set_zlabel("Player 3 confidence (PGD)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.view_init(elev=25, azim=-45)
ax.legend()
plt.title("Asymmetric Learning Dynamics: MWU vs PGD\n(Separatrix generated instantly via exact formula)")
plt.grid(True)
plt.show()