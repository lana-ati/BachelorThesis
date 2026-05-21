import math
import random


def mwu_step(score_A, score_B, opponent_conf_A, learning_speed, noise_A, noise_B):
    r_A = opponent_conf_A + noise_A
    r_B = (1 - opponent_conf_A) + noise_B
    return score_A + learning_speed * r_A, score_B + learning_speed * r_B