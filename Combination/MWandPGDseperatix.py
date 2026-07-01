import math
import random
import matplotlib.pyplot as plt
from mwu_sim import mwu_step
from pgd_sim import pgd_step


def run_mixed_simulation(p1_start, p2_start, payoff_matrix_p1, payoff_matrix_p2, iterations=1000, noise = 0.2, ):
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
        next_p2_conf = pgd_step(p2_conf, p1_conf, learning_speed, payoff_matrix_p2, noise_A_2, noise_B_2)

        # 2. Compute MWU's next scores based on PGD's current confidence
        next_p1_score_A, next_p1_score_B = mwu_step(p1_score_A, p1_score_B, p2_conf, learning_speed, payoff_matrix_p1,
                                                    noise_A_1, noise_B_1)

        p2_conf = next_p2_conf
        p1_score_A = next_p1_score_A
        p1_score_B = next_p1_score_B

    return history
#Graphics!!

# Recreate your anti-diagonal initial scenarios
test_scenarios = []
for i in range(100):
    p1 = random.uniform(0.01, 0.99)
    p2 = random.uniform(0.01, 0.99)
    test_scenarios.append((p1, p2))

# Plotting Loop
plt.figure(figsize=(8, 8))

payoff_matrix_p1 = [
    [1, 0],
    [0, 1]
]

payoff_matrix_p2 = [
    [1, 0],
    [0, 1]
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


def check_convergence_destination(p1_start, p2_start):
    history = run_mixed_simulation(p1_start, p2_start, payoff_matrix_p1, payoff_matrix_p2, iterations=200, noise=0)
    final_p1, final_p2 = history[-1]

    # Return 1 if it goes to the top-right, 0 if it goes to the bottom-left
    if final_p1 > 0.5 and final_p2 > 0.5:
        return 1
    return 0


def find_separatrix_y(p1_coordinate, tolerance=1e-5):
    """Uses binary search to find the precise y-boundary for a given x."""
    low_y = 0.0
    high_y = 1.0

    while (high_y - low_y) > tolerance:
        mid_y = (low_y + high_y) / 2
        destination = check_convergence_destination(p1_coordinate, mid_y)

        # If it went to (1,1), the boundary is lower down
        if destination == 1:
            high_y = mid_y
        # If it went to (0,0), the boundary is higher up
        else:
            low_y = mid_y

    return (low_y + high_y) / 2

# Step 1: Find the y-boundary at x = 0.4
x1 = 0.4
y1 = find_separatrix_y(x1)

# Step 2: Find the y-boundary at x = 0.6
x2 = 0.6
y2 = find_separatrix_y(x2)

# Step 3: Compute Gradient (m = change in y / change in x)
gradient = (y2 - y1) / (x2 - x1)

print(f"Point 1 on Separatrix: ({x1}, {y1:.5f})")
print(f"Point 2 on Separatrix: ({x2}, {y2:.5f})")
print(f"Calculated Gradient (Slope) of the line: {gradient:.4f}")


line_x = [0.0, 1.0]
line_y = [gradient * (x - 0.5) + 0.5 for x in line_x]
plt.plot(line_x, line_y, color="red", linestyle="--", linewidth=2, zorder=4,
         label=f"Straight Separatrix (Slope: {gradient:.2f})")


plt.show()