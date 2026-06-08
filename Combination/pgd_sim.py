
def pgd_step(conf_A, opponent_conf_A, learning_speed, payoff_matrix, noise_A=0, noise_B=0):

    pA = opponent_conf_A
    pB = 1 - opponent_conf_A

    # Expected payoff of choosing A
    r_A = (
        pA * payoff_matrix[0][0] +
        pB * payoff_matrix[0][1]
    ) + noise_A

    # Expected payoff of choosing B
    r_B = (
        pA * payoff_matrix[1][0] +
        pB * payoff_matrix[1][1]
    ) + noise_B

    gradient = r_A - r_B

    return max(0, min(1, conf_A + learning_speed * gradient))