import random
import math
import matplotlib.pyplot as plt

def simulate_pgd(p1_start_chance, p2_start_chance,  payoff_matrix_p1, payoff_matrix_p2, learning_rate=0.1, iterations=1000):
    noise_level = 0.2

    p1_confidence_A = p1_start_chance
    p2_confidence_A = p2_start_chance

    plot_history = []

    for i in range(iterations):

        #robbins monro step size
        learning_speed = learning_rate / math.sqrt(i + 1)

        p1_confidence_B = 1 - p1_confidence_A
        p2_confidence_B = 1 - p2_confidence_A

        plot_history.append((p1_confidence_A, p2_confidence_A))

        #calculate rewards
        # calculate expected rewards

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

        #gradient update
        p1_confidence_A += learning_speed * (reward_1_A - reward_1_B)
        p2_confidence_A += learning_speed * (reward_2_A - reward_2_B)

        #projection step
        p1_confidence_A = max(0, min(1, p1_confidence_A))
        p2_confidence_A = max(0, min(1, p2_confidence_A))

    return plot_history


test_scenarios = []
for i in range(40):
    p1 = random.uniform(0.01, 0.99)
    p2 = random.uniform(0.01, 0.99)
    test_scenarios.append((p1, p2))


payoff_matrix_p1 = [
    [1, 0],
    [0, 1]
]

payoff_matrix_p2 = [
    [1, 0],
    [0, 1]
]

#Graphics!
for p1, p2 in test_scenarios:
    history = simulate_pgd(p1, p2, payoff_matrix_p1, payoff_matrix_p2)

    x = [h[0] for h in history]
    y = [h[1] for h in history]

    plt.plot(x, y, color="navy", linewidth=1)
    plt.scatter(p1, p2, color='black', s=15, zorder=3)


plt.xlabel("Player 1 confidence in A")
plt.ylabel("Player 2 confidence in A")

plt.title("Coordination Game, both players using Projected Gradient Descent")

plt.xlim(0,1)
plt.ylim(0,1)

plt.scatter([0.5], [0.5], color='red', s=40)

plt.grid(True)
plt.show()