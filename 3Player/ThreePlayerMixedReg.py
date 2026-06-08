import random
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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
    iterations=1000
):
    noise_level = 0.1

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


# Generate random 3D test scenarios
test_scenarios = []
for i in range(100):
    p1 = random.uniform(0.01, 0.99)
    p2 = random.uniform(0.01, 0.99)
    p3 = random.uniform(0.01, 0.99)
    test_scenarios.append((p1, p2, p3))

# Graphics Setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

payoff_matrix_p1 = [
    [1, 0],
    [0, 1]
]

payoff_matrix_p2 = payoff_matrix_p1
payoff_matrix_p3 = payoff_matrix_p1

for p1, p2, p3 in test_scenarios:
    history = simulate_mixed_game(p1, p2, p3, payoff_matrix_p1, payoff_matrix_p2, payoff_matrix_p3)

    x = [h[0] for h in history]
    y = [h[1] for h in history]
    z = [h[2] for h in history]

    ax.plot(x, y, z, color="navy", linewidth=1, alpha=0.6)
    ax.scatter(p1, p2, p3, color='black', s=15, zorder=3)

# Labeling axes for distinct player types
ax.set_xlabel("Player 1 confidence (MWU)")
ax.set_ylabel("Player 2 confidence (PGD)")
ax.set_zlabel("Player 3 confidence (PGD)")

plt.title("Mixed Learning Dynamics: MWU (P1) vs PGD (P2 & P3)")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

# Unstable equilibrium point at the center
ax.scatter([0.5], [0.5], [0.5], color='red', s=50, label='Unstable Equilibrium')


# Calculate Separatrix geometry
def calculate_skewed_separatrix(A, B, C, D):
    """
    Dynamically finds the intersection points of the plane Ax + By + Cz = D
    within the unit cube [0, 1]^3.
    """
    points = []

    # Check 4 edges parallel to X-axis (y and z fixed)
    for y in [0, 1]:
        for z in [0, 1]:
            if A != 0:
                x = (D - B * y - C * z) / A
                if 0 <= x <= 1: points.append([x, y, z])

    # Check 4 edges parallel to Y-axis (x and z fixed)
    for x in [0, 1]:
        for z in [0, 1]:
            if B != 0:
                y = (D - A * x - C * z) / B
                if 0 <= y <= 1: points.append([x, y, z])

    # Check 4 edges parallel to Z-axis (x and y fixed)
    for x in [0, 1]:
        for y in [0, 1]:
            if C != 0:
                z = (D - A * x - B * y) / C
                if 0 <= z <= 1: points.append([x, y, z])

    # Filter duplicates (using rounding to prevent floating point edge cases)
    unique_points = np.unique(np.round(points, decimals=5), axis=0)

    if len(unique_points) < 3:
        return []

    # Sort vertices angularly around their centroid
    centroid = np.mean(unique_points, axis=0)

    # Generate an arbitrary 2D basis on the plane
    normal = np.array([A, B, C])
    normal = normal / np.linalg.norm(normal)

    # Find a vector not parallel to normal to generate orthogonal basis
    if abs(normal[0]) < 0.9:
        v1 = np.cross(normal, [1, 0, 0])
    else:
        v1 = np.cross(normal, [0, 1, 0])

    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)

    angles = [np.arctan2(np.dot(p - centroid, v2), np.dot(p - centroid, v1)) for p in unique_points]
    return unique_points[np.argsort(angles)].tolist()


# The mathematically derived coefficients for the mixed dynamic
A = 4
B = 1 + math.sqrt(3)
C = 1 + math.sqrt(3)
D = 3 + math.sqrt(3)

calculated_vertices = calculate_skewed_separatrix(A, B, C, D)

if calculated_vertices:
    separatrix = Poly3DCollection([calculated_vertices], alpha=0.35, facecolors='mediumseagreen',
                                  edgecolors='darkgreen', linewidths=1.5)
    separatrix.set_label('True Mixed Separatrix')
    ax.add_collection3d(separatrix)


ax.view_init(elev=25, azim=-45)
ax.legend()
plt.grid(True)
plt.show()