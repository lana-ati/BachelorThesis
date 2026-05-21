import random

def pgd_step(conf_A, opponent_conf_A, learning_speed, noise_A, noise_B):
    r_A = opponent_conf_A + noise_A
    r_B = (1 - opponent_conf_A) + noise_B
    gradient = r_A - r_B
    return max(0, min(1, conf_A + learning_speed * gradient))