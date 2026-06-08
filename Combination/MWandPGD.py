import math
import random
import matplotlib.pyplot as plt

from mwu_sim import mwu_step
from pgd_sim import pgd_step


def run_mixed_simulation(p1_start, p2_start, payoff_matrix_p1, payoff_matrix_p2, iterations=1000, noise=0.2):

    # Initialize Player 1 (MWU) internal log-scores
    p1_score_A = math.log(p1_start)
    p1_score_B = math.log(1 - p1_start)

    # Initialize Player 2 (PGD) direct probability
    p2_conf = p2_start

    history = []

    for i in range(iterations):
        learning_speed = 0.1 / math.sqrt(i + 1)

        # Calculate P1's current probability before updating states
        denom = math.exp(p1_score_A) + math.exp(p1_score_B)
        p1_conf = math.exp(p1_score_A) / denom

        # Record the current state of play
        history.append((p1_conf, p2_conf))

        noise_A_1 = random.uniform(0, noise)
        noise_B_1 = random.uniform(0, noise)

        noise_A_2 = random.uniform(0, noise)
        noise_B_2 = random.uniform(0, noise)

        # 1. Compute PGD's next move based on MWU's current confidence
        next_p2_conf = pgd_step(p2_conf, p1_conf, learning_speed, payoff_matrix_p2, noise_A_2,noise_B_2)

        # 2. Compute MWU's next scores based on PGD's current confidence
        next_p1_score_A, next_p1_score_B = mwu_step(p1_score_A, p1_score_B, p2_conf, learning_speed, payoff_matrix_p1, noise_A_1, noise_B_1)

        p2_conf = next_p2_conf
        p1_score_A = next_p1_score_A
        p1_score_B = next_p1_score_B

    return history

# ==========================================
# GRAPHICS GENERATION
# ==========================================

# Recreate your anti-diagonal initial scenarios
test_scenarios = []
for i in range(100):
    p1 = random.uniform(0.01, 0.99)
    p2 = random.uniform(0.01, 0.99)
    test_scenarios.append((p1, p2))

# Plotting Loop
plt.figure(figsize=(8, 8))

payoff_matrix_p1 = [
    [2, 0],
    [0, 3]
]

payoff_matrix_p2 = [
    [2, 0],
    [0, 3]
]


for p1, p2 in test_scenarios:
    history = run_mixed_simulation(p1, p2, payoff_matrix_p1, payoff_matrix_p2)

    x_coords = [h[0] for h in history]
    y_coords = [h[1] for h in history]

    # Draw the trajectory line
    plt.plot(x_coords, y_coords, color="navy", linewidth=1, alpha=0.6)
    # Mark the initial starting point
    plt.scatter(p1, p2, color='black', s=15, zorder=3)

# Formatting the chart
plt.xlabel("Player 1 confidence in A (MWU)")
plt.ylabel("Player 2 confidence in A (PGD)")
plt.title("Coordination Game: Asymmetric Dynamics (MWU vs PGD)")

plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)

# Highlight center unstable Nash Equilibrium
plt.scatter([0.5], [0.5], color='red', s=40, zorder=4, label="Unstable Equilibrium")

plt.grid(True)
plt.legend()

plt.show()

