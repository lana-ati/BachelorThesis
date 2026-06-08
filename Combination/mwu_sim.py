

def mwu_step(score_A, score_B, opponent_conf_A, learning_speed, payoff_matrix, noise_A=0, noise_B=0):

    pA = opponent_conf_A
    pB = 1 - opponent_conf_A

    # Expected payoff of action A
    r_A = (
        pA * payoff_matrix[0][0] +
        pB * payoff_matrix[0][1]
    ) + noise_A

    # Expected payoff of action B
    r_B = (
        pA * payoff_matrix[1][0] +
        pB * payoff_matrix[1][1]
    ) + noise_B

    return (
        score_A + learning_speed * r_A,
        score_B + learning_speed * r_B
    )