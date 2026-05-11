import math
import random

def simulate_mwu(p1_start_chance, p2_start_chance, iterations=1000):
    learning_speed = 0.1
    noise_level = 0.2

    #init scores
    p1_score_A = math.log(p1_start_chance)
    p1_score_B = math.log(1 - p1_start_chance)

    p2_score_A = math.log(p2_start_chance)
    p2_score_B = math.log(1 - p2_start_chance)

    for i in range(iterations):

        #confidence p1
        denom1 = math.exp(p1_score_A) + math.exp(p1_score_B)
        p1_confidence_A = math.exp(p1_score_A) / denom1
        p1_confidence_B = 1 - p1_confidence_A

        #confidence p2
        denom2 = math.exp(p2_score_A) + math.exp(p2_score_B)
        p2_confidence_A = math.exp(p2_score_A) / denom2
        p2_confidence_B = 1 - p2_confidence_A

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


    return round(p1_confidence_A, 2), round(p2_confidence_A, 2)


test_scenarios = [
    (0.1, 0.1), (0.1, 0.4), (0.1, 0.7), (0.1, 0.9),
    (0.4, 0.1), (0.7, 0.1), (0.9, 0.1),
    (0.4, 0.9), (0.7, 0.9), (0.9, 0.9),
    (0.9, 0.4), (0.9, 0.7),
    (0.45, 0.55), (0.55, 0.45)  # Points near the saddle
]


print(f"{'Starting Confidence':<30} | {'Final Confidence':<20} | Outcome")
print("-" * 70)

for p1, p2 in test_scenarios:
    final_p1, final_p2 = simulate_mwu(p1, p2)

    #print results: readable
    if final_p1 > 0.5 and final_p2 > 0.5:
        result = "Both chose A"
    elif final_p1 < 0.5 and final_p2 < 0.5:
        result = "Both chose B"
    else:
        result = "Confusion"

    print(
        f"P1: {p1 * 100:>2}%, P2: {p2 * 100:>2}%{'':<8} | P1: {final_p1 * 100:>3}%, P2: {final_p2 * 100:>3}% | {result}")