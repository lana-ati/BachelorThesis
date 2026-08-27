import math
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

def simulate_mwu(p1_start_chance, p2_start_chance, payoff_matrix_p1, payoff_matrix_p2, learning_rate=0.1, noise_level = 0.2, iterations=5000):

    #init scores
    p1_score_A = math.log(p1_start_chance)
    p1_score_B = math.log(1 - p1_start_chance)

    p2_score_A = math.log(p2_start_chance)
    p2_score_B = math.log(1 - p2_start_chance)

    plot_history = []

    for i in range(iterations):

        #robbins monro step size
        learning_speed = learning_rate / math.sqrt(i + 1)

        #confidence p1
        denom1 = math.exp(p1_score_A) + math.exp(p1_score_B)
        p1_confidence_A = math.exp(p1_score_A) / denom1
        p1_confidence_B = 1 - p1_confidence_A

        #confidence p2
        denom2 = math.exp(p2_score_A) + math.exp(p2_score_B)
        p2_confidence_A = math.exp(p2_score_A) / denom2
        p2_confidence_B = 1 - p2_confidence_A

        plot_history.append((p1_confidence_A, p2_confidence_A))

        #calculate rewards
        reward_1_A = (
                payoff_matrix_p1[0][0] * p2_confidence_A +
                payoff_matrix_p1[0][1] * p2_confidence_B
        )

        reward_1_B = (
                payoff_matrix_p1[1][0] * p2_confidence_A +
                payoff_matrix_p1[1][1] * p2_confidence_B
        )

        reward_2_A = (
                payoff_matrix_p2[0][0] * p1_confidence_A +
                payoff_matrix_p2[1][0] * p1_confidence_B
        )

        reward_2_B = (
                payoff_matrix_p2[0][1] * p1_confidence_A +
                payoff_matrix_p2[1][1] * p1_confidence_B
        )

        #noise
        reward_1_A += random.uniform(-noise_level, noise_level)
        reward_1_B += random.uniform(-noise_level, noise_level)
        reward_2_A += random.uniform(-noise_level, noise_level)
        reward_2_B += random.uniform(-noise_level, noise_level)

        #update score
        p1_score_A += learning_speed * reward_1_A
        p1_score_B += learning_speed * reward_1_B

        p2_score_A += learning_speed * reward_2_A
        p2_score_B += learning_speed * reward_2_B


    #return p1_confidence_A, p2_confidence_A
    return plot_history

test_scenarios = []
for i in range(100):
    p1 = random.uniform(0.01, 0.99)
    p2 = random.uniform(0.01, 0.99)
    test_scenarios.append((p1, p2))

# Initial Default Parameters
init_lr = 0.1
init_noise = 0.2

payoff_matrix_p1 = [
    [1, 0],
    [0, 1]
]
payoff_matrix_p2 = [
    [1, 0],
    [0, 1]
]
def mixed_equilibrium(matrix_p1, matrix_p2):
    a1, b1 = matrix_p1[0]
    c1, d1 = matrix_p1[1]

    a2, b2 = matrix_p2[0]
    c2, d2 = matrix_p2[1]

    denom1 = a2 - b2 - c2 + d2
    denom2 = a1 - b1 - c1 + d1

    if abs(denom1) < 1e-12 or abs(denom2) < 1e-12:
        return None

    p1 = (d2 - c2) / denom1
    p2 = (d1 - b1) / denom2

    return p1, p2



#Graphics!
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.35, left=0.25)

def redraw_simulation(val=None):
    """Clears the axes and redraws the simulation trajectories based on current UI values."""
    ax.clear()

    # Fetch parameters from sliders
    lr = slider_lr.val
    noise = slider_noise.val

    # Safely parse payoff matrices from the text boxes
    try:
        m1 = eval(text_p1.text)
        m2 = eval(text_p2.text)
    except Exception:
        # Fallback to default if text entry is invalid while typing
        m1, m2 = payoff_matrix_p1, payoff_matrix_p2

    # Run simulations and plot
    for p1, p2 in test_scenarios:
        history = simulate_mwu(p1, p2, m1, m2, learning_rate=lr, noise_level=noise)
        x = [h[0] for h in history]
        y = [h[1] for h in history]
        ax.plot(x, y, color="navy", linewidth=1, alpha=0.6)
        ax.scatter(p1, p2, color='black', s=15, zorder=3)

    # Plot formatting
    ax.set_xlabel("Player 1 confidence in A")
    ax.set_ylabel("Player 2 confidence in A")
    ax.set_title("Coordination Game (Multiplicative Weights)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect('equal', adjustable='box')
    c = mixed_equilibrium(m1, m2)

    if c is not None:
        c1, c2 = c

        # plot equilibrium
        ax.scatter(c1, c2, color="red", s=40, zorder=4,
                   label="Unstable Equilibrium")

        # separatrix with slope -1 through c
        x_sep = [0, 1]
        y_sep = [(c1 + c2) - x for x in x_sep]

        ax.plot(
            x_sep,
            y_sep,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Separatrix"
        )
    ax.grid(True)
    fig.canvas.draw_idle()

    # Count convergence destinations
    converged_A = 0
    converged_B = 0

    for p1, p2 in test_scenarios:
        history = simulate_mwu(
            p1, p2,
            m1, m2,
            learning_rate=lr,
            noise_level=0,  # deterministic basin measurement
            iterations=5000
        )

        final_p1, final_p2 = history[-1]

        # (1,1) equilibrium = A
        if final_p1 > 0.5 and final_p2 > 0.5:
            converged_A += 1
        # (0,0) equilibrium = B
        elif final_p1 < 0.5 and final_p2 < 0.5:
            converged_B += 1

    total = len(test_scenarios)

    percentage_A = (converged_A / total) * 100
    percentage_B = (converged_B / total) * 100

    print(f"Convergence to A (1,1): {converged_A}/{total} ({percentage_A:.2f}%)")
    print(f"Convergence to B (0,0): {converged_B}/{total} ({percentage_B:.2f}%)")

# UI Layout controls

ax_lr = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_noise = plt.axes([0.25, 0.15, 0.65, 0.03])

slider_lr = Slider(ax_lr, 'Learning Rate', 0.01, 1.0, valinit=init_lr, valstep=0.01)
slider_noise = Slider(ax_noise, 'Noise Level', 0.0, 1.0, valinit=init_noise, valstep=0.01)

# Text Boxes for matrices
ax_p1 = plt.axes([0.25, 0.08, 0.25, 0.04])
ax_p2 = plt.axes([0.65, 0.08, 0.25, 0.04])

text_p1 = TextBox(ax_p1, 'P1 Matrix ', initial=str(payoff_matrix_p1))
text_p2 = TextBox(ax_p2, 'P2 Matrix ', initial=str(payoff_matrix_p2))

# Bind update events to UI components
slider_lr.on_changed(redraw_simulation)
slider_noise.on_changed(redraw_simulation)
text_p1.on_submit(redraw_simulation)
text_p2.on_submit(redraw_simulation)

# Initial draw
redraw_simulation()




ax.legend()
plt.show()