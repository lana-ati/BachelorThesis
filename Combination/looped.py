import math
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from mwu_sim import mwu_step
from pgd_sim import pgd_step


def run_mixed_simulation(p1_start, p2_start, iterations=1000, noise=0.1):
    p1_score_A = math.log(p1_start)
    p1_score_B = math.log(1 - p1_start)
    p2_conf = p2_start

    history = []

    for i in range(iterations):
        learning_speed = 0.1 / math.sqrt(i + 1)

        denom = math.exp(p1_score_A) + math.exp(p1_score_B)
        p1_conf = math.exp(p1_score_A) / denom

        history.append((p1_conf, p2_conf))

        noise_A_1 = random.uniform(0, noise)
        noise_B_1 = random.uniform(0, noise)
        noise_A_2 = random.uniform(0, noise)
        noise_B_2 = random.uniform(0, noise)

        next_p2_conf = pgd_step(p2_conf, p1_conf, learning_speed, noise_A_2, noise_B_2)
        next_p1_score_A, next_p1_score_B = mwu_step(
            p1_score_A, p1_score_B, p2_conf,
            learning_speed, noise_A_1, noise_B_1
        )

        p2_conf = next_p2_conf
        p1_score_A = next_p1_score_A
        p1_score_B = next_p1_score_B

    return history

def check_convergence_destination(p1_start, p2_start):
    history = run_mixed_simulation(p1_start, p2_start, iterations=200)
    final_p1, final_p2 = history[-1]

    return 1 if (final_p1 > 0.5 and final_p2 > 0.5) else 0


def find_separatrix_y(p1_coordinate, tolerance=1e-5):
    low_y, high_y = 0.0, 1.0

    while (high_y - low_y) > tolerance:
        mid_y = (low_y + high_y) / 2
        destination = check_convergence_destination(p1_coordinate, mid_y)

        if destination == 1:
            high_y = mid_y
        else:
            low_y = mid_y

    return (low_y + high_y) / 2

fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.2)


def run_and_plot(event=None):
    ax.clear()

    """test_scenarios = []
    for i in range(40):
        p1 = random.uniform(0.01, 0.99)
        p2 = random.uniform(0.01, 0.99)
        test_scenarios.append((p1, p2))"""

    test_scenarios = []
    for i in range(40):
        p1 = random.uniform(0, 1)
        x = random.uniform(-0.02, 0.02)

        p2 = 1 - p1 + x

        p2 = max(0, min(1, p2))

        test_scenarios.append((p1, p2))


    for p1, p2 in test_scenarios:
        history = run_mixed_simulation(p1, p2)

        x_coords = [h[0] for h in history]
        y_coords = [h[1] for h in history]

        ax.plot(x_coords, y_coords, color="navy", linewidth=1, alpha=0.5)
        ax.scatter(p1, p2, color="black", s=15)

    x1, x2 = 0.4, 0.6
    y1 = find_separatrix_y(x1)
    y2 = find_separatrix_y(x2)

    gradient = (y2 - y1) / (x2 - x1)

    print(f"Gradient: {gradient:.4f}")

    line_x = [0.0, 1.0]
    line_y = [gradient * (x - 0.5) + 0.5 for x in line_x]

    ax.plot(
        line_x, line_y,
        color="red", linestyle="--", linewidth=2,
        label=f"Separatrix slope: {gradient:.2f}"
    )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    ax.set_xlabel("Player 1 confidence (MWU)")
    ax.set_ylabel("Player 2 confidence (PGD)")
    ax.set_title("MWU vs PGD Dynamics (Redo button)")

    ax.scatter([0.5], [0.5], color="red", s=50, label="Unstable equilibrium")

    ax.grid(True)
    ax.legend()

    fig.canvas.draw_idle()


button_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
button = Button(button_ax, "Redo")

button.on_clicked(run_and_plot)

# initial draw
run_and_plot()

plt.show()