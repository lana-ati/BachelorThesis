import math
import random
import matplotlib.pyplot as plt

def simulate_mwu(p1_start_chance, p2_start_chance, iterations=1000):
    learning_speed = 0.1
    noise_level = 0.2

    #init scores
    p1_score_A = math.log(p1_start_chance)
    p1_score_B = math.log(1 - p1_start_chance)

    p2_score_A = math.log(p2_start_chance)
    p2_score_B = math.log(1 - p2_start_chance)
    plot_history = []

    for i in range(iterations):

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
        reward_1_A = 2 * p2_confidence_A
        reward_1_B = 2 * p2_confidence_B

        reward_2_A = 2 * p1_confidence_A
        reward_2_B = 2 * p1_confidence_B

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


test_scenarios = [
    (0.1, 0.1), (0.1, 0.4), (0.1, 0.7), (0.1, 0.9),
    (0.4, 0.1), (0.7, 0.1), (0.9, 0.1),
    (0.4, 0.9), (0.7, 0.9), (0.9, 0.9),
    (0.9, 0.4), (0.9, 0.7),
    (0.45, 0.55), (0.55, 0.45)
]


print(f"{'Starting Confidence':<30} | {'Final Confidence':<20} | Outcome")
print("-" * 70)

"""for p1, p2 in test_scenarios:
    final_p1, final_p2 = simulate_mwu(p1, p2)
    p1_rounded = round(final_p1, 2)
    p2_rounded = round(final_p2, 2)

    print (f"final p1 {final_p1}, final p2 {final_p2}")
    print (f"final rounded {p1_rounded}, final rounded {p2_rounded}")"""


#Graphics!
for p1, p2 in test_scenarios:
    history = simulate_mwu(p1, p2)

    x = [h[0] for h in history]
    y = [h[1] for h in history]

    plt.plot(x, y)


plt.xlabel("Player 1 confidence in A")
plt.ylabel("Player 2 confidence in A")

plt.title("Coordination Game, both players using Multiplicative Weights")

plt.xlim(0,1)
plt.ylim(0,1)

plt.grid(True)
plt.show()