import math
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

from mwu_sim import mwu_step
from pgd_sim import pgd_step


def run_mixed_simulation(p1_start, p2_start, payoff_matrix_p1, payoff_matrix_p2, iterations=1000, learning_rate = 0.1, noise=0.2):

    # Initialize Player 1 (MWU) internal log-scores
    p1_score_A = math.log(p1_start)
    p1_score_B = math.log(1 - p1_start)

    # Initialize Player 2 (PGD) direct probability
    p2_conf = p2_start

    history = []

    for i in range(iterations):
        learning_speed = learning_rate / math.sqrt(i + 1)

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
        p2_conf = pgd_step(
            p2_conf,
            p1_conf,
            learning_speed,
            payoff_matrix_p2,
            noise_A_2,
            noise_B_2,
        )

        p1_score_A, p1_score_B = mwu_step(
            p1_score_A,
            p1_score_B,
            p2_conf,
            learning_speed,
            payoff_matrix_p1,
            noise_A_1,
            noise_B_1,
        )

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

fig, ax = plt.subplots(figsize=(8,8))
plt.subplots_adjust(bottom=0.28)

init_lr = 0.1
init_noise = 0.2

def redraw_simulation(val=None):

    ax.clear()

    lr = slider_lr.val
    noise = slider_noise.val

    for p1, p2 in test_scenarios:

        history = run_mixed_simulation(
            p1,
            p2,
            payoff_matrix_p1,
            payoff_matrix_p2,
            learning_rate=lr,
            noise=noise
        )

        xs = [h[0] for h in history]
        ys = [h[1] for h in history]

        ax.plot(xs, ys, color="navy", linewidth=1, alpha=0.6)
        ax.scatter(p1, p2, color="black", s=15)

    a = payoff_matrix_p1[0][0]
    b = payoff_matrix_p2[1][1]

    gradient = -(a+b)/math.sqrt(a*b)
    c = b/(a+b)

    line_x = [0,1]
    line_y = [gradient*(x-c)+c for x in line_x]

    ax.plot(
        line_x,
        line_y,
        "r--",
        linewidth=2,
        label=f"Separatrix ({gradient:.2f})"
    )

    ax.scatter(
        [0.5],
        [0.5],
        color="red",
        s=40,
        label="Unstable Equilibrium"
    )

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    ax.set_xlabel("Player 1 confidence in A (MWU)")
    ax.set_ylabel("Player 2 confidence in A (PGD)")
    ax.set_title("Coordination Game: MWU vs PGD")
    ax.grid(True)
    ax.legend()

    fig.canvas.draw_idle()

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

a = payoff_matrix_p1[0][0]
b = payoff_matrix_p2[1][1]
gradient = -(a+b) / math.sqrt(a*b)

c = b/ (a+b)

line_x = [0.0, 1.0]
line_y = [gradient * (x - c) + c for x in line_x]
plt.plot(line_x, line_y, color="red", linestyle="--", linewidth=2, zorder=4,
         label=f"Straight Separatrix (Slope: {gradient:.2f})")

plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)

intercept = c - gradient * c

print(
    f"{gradient:.4f}x - 1.0000y = {-intercept:.4f}"
)

# Highlight center unstable Nash Equilibrium
plt.scatter([0.5], [0.5], color='red', s=40, zorder=4, label="Unstable Equilibrium")


ax_lr = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_noise = plt.axes([0.25, 0.15, 0.65, 0.03])


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

ax_p1 = plt.axes([0.25, 0.08, 0.25, 0.04])
ax_p2 = plt.axes([0.65, 0.08, 0.25, 0.04])

text_p1 = TextBox(ax_p1, 'P1 Matrix ', initial=str(payoff_matrix_p1))
text_p2 = TextBox(ax_p2, 'P2 Matrix ', initial=str(payoff_matrix_p2))

slider_lr.on_changed(redraw_simulation)
slider_noise.on_changed(redraw_simulation)





plt.grid(True)

redraw_simulation()
plt.show()