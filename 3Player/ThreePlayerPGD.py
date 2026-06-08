import random
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def simulate_pgd(p1_start_chance, p2_start_chance, p3_start_chance, matrix_p1, matrix_p2, matrix_p3, learning_rate=0.1, iterations=1000):
    noise_level = 0.1

    p1_confidence_A = p1_start_chance
    p2_confidence_A = p2_start_chance
    p3_confidence_A = p3_start_chance

    plot_history = []

    for i in range(iterations):

        # Robbins-Monro step size
        learning_speed = learning_rate / math.sqrt(i + 1)

        # p(B) = 1 - p(A)
        p1_confidence_B = 1 - p1_confidence_A
        p2_confidence_B = 1 - p2_confidence_A
        p3_confidence_B = 1 - p3_confidence_A

        plot_history.append((p1_confidence_A, p2_confidence_A, p3_confidence_A))

        # Calculate rewards
        # (Requires all three players to coordinate for a payout)
        reward_1_A = (p2_confidence_A * p3_confidence_A) * matrix_p1[0][0] + (p2_confidence_B * p3_confidence_B) * \
                     matrix_p1[0][1]
        reward_1_B = (p2_confidence_A * p3_confidence_A) * matrix_p1[1][0] + (p2_confidence_B * p3_confidence_B) * \
                     matrix_p1[1][1]

        reward_2_A = (p1_confidence_A * p3_confidence_A) * matrix_p2[0][0] + (p1_confidence_B * p3_confidence_B) * \
                     matrix_p2[0][1]
        reward_2_B = (p1_confidence_A * p3_confidence_A) * matrix_p2[1][0] + (p1_confidence_B * p3_confidence_B) * \
                     matrix_p2[1][1]

        reward_3_A = (p1_confidence_A * p2_confidence_A) * matrix_p3[0][0] + (p1_confidence_B * p2_confidence_B) * \
                     matrix_p3[0][1]
        reward_3_B = (p1_confidence_A * p2_confidence_A) * matrix_p3[1][0] + (p1_confidence_B * p2_confidence_B) * \
                     matrix_p3[1][1]

        # Noise
        reward_1_A += random.uniform(-noise_level, noise_level)
        reward_1_B += random.uniform(-noise_level, noise_level)
        reward_2_A += random.uniform(-noise_level, noise_level)
        reward_2_B += random.uniform(-noise_level, noise_level)
        reward_3_A += random.uniform(-noise_level, noise_level)
        reward_3_B += random.uniform(-noise_level, noise_level)

        # Gradient update
        p1_confidence_A += learning_speed * (reward_1_A - reward_1_B)
        p2_confidence_A += learning_speed * (reward_2_A - reward_2_B)
        p3_confidence_A += learning_speed * (reward_3_A - reward_3_B)

        # Projection step back into [0,1]
        p1_confidence_A = max(0, min(1, p1_confidence_A))
        p2_confidence_A = max(0, min(1, p2_confidence_A))
        p3_confidence_A = max(0, min(1, p3_confidence_A))

        # Break if all players have completely converged to a consensus
        if ((p1_confidence_A == 1 and p2_confidence_A == 1 and p3_confidence_A == 1) or
                (p1_confidence_A == 0 and p2_confidence_A == 0 and p3_confidence_A == 0)):
            break

    return plot_history

payoff_matrix_p1 = [
    [1, 0],
    [0, 1]
]

payoff_matrix_p2 = payoff_matrix_p1
payoff_matrix_p3 = payoff_matrix_p1

# Generate random 3D test scenarios
test_scenarios = []
for i in range(40):
    p1 = random.uniform(0, 1)
    p2 = random.uniform(0, 1)
    p3 = random.uniform(0, 1)
    test_scenarios.append((p1, p2, p3))

# Graphics Setup
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for p1, p2, p3 in test_scenarios:
    history = simulate_pgd(p1, p2, p3, payoff_matrix_p1, payoff_matrix_p2, payoff_matrix_p3)

    x = [h[0] for h in history]
    y = [h[1] for h in history]
    z = [h[2] for h in history]

    ax.plot(x, y, z, color="navy", linewidth=1)
    ax.scatter(p1, p2, p3, color='black', s=15, zorder=3)

# Labeling axes for 3 players
ax.set_xlabel("Player 1 confidence in A")
ax.set_ylabel("Player 2 confidence in A")
ax.set_zlabel("Player 3 confidence in A")

plt.title("3-Player Coordination Game using Projected Gradient Descent")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

# Unstable equilibrium point at the center (0.5, 0.5, 0.5)
ax.scatter([0.5], [0.5], [0.5], color='red', s=50)

#seperatix?
def calculate_separatrix_vertices(c=1.5):
    """
    Dynamically finds and sorts the intersection points of the plane
    x + y + z = c within the unit cube [0, 1]^3.
    """
    points = []

    # Check 4 edges parallel to X-axis (y and z fixed at 0 or 1)
    for y in [0, 1]:
        for z in [0, 1]:
            x = c - y - z
            if 0 <= x <= 1: points.append([x, y, z])

    # Check 4 edges parallel to Y-axis (x and z fixed at 0 or 1)
    for x in [0, 1]:
        for z in [0, 1]:
            y = c - x - z
            if 0 <= y <= 1: points.append([x, y, z])

    # Check 4 edges parallel to Z-axis (x and y fixed at 0 or 1)
    for x in [0, 1]:
        for y in [0, 1]:
            z = c - x - y
            if 0 <= z <= 1: points.append([x, y, z])

    # Filter out any duplicate corners
    unique_points = np.unique(points, axis=0)

    if len(unique_points) < 3:
        return []

    # Sort vertices angularly around their centroid so the polygon renders cleanly
    centroid = np.mean(unique_points, axis=0)

    # Define a 2D coordinate system on the plane (Normal vector is [1, 1, 1])
    normal = np.array([1.0, 1.0, 1.0])
    v1 = np.array([1.0, -1.0, 0.0])  # Perpendicular to normal
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)  # Perpendicular to both
    v2 /= np.linalg.norm(v2)

    # Project 3D points to 2D plane angles relative to the centroid
    angles = []
    for p in unique_points:
        vector_from_center = p - centroid
        x_proj = np.dot(vector_from_center, v1)
        y_proj = np.dot(vector_from_center, v2)
        angles.append(np.arctan2(y_proj, x_proj))

    # Sort points using the computed angles
    sorted_indices = np.argsort(angles)
    return unique_points[sorted_indices].tolist()

calculated_vertices = calculate_separatrix_vertices(c=1.5)

if calculated_vertices:
    separatrix = Poly3DCollection([calculated_vertices], alpha=0.25, facecolors='crimson', edgecolors='darkred', linewidths=1.5)
    separatrix.set_label('Calculated Separatrix (x+y+z=1.5)')
    ax.add_collection3d(separatrix)


plt.grid(True)
plt.show()