import math

# payoff matrices
payoff_p1 = [
    [3, 0],
    [0, 2]
]

payoff_p2 = [
    [3, 0],
    [0, 2]
]

# Initial weights
weights_p1 = [1, 1]
weights_p2 = [1, 1]

# Learning rate
eta = 0.1


# STEP 1: Convert weights into probabilities
sum_p1 = sum(weights_p1)
sum_p2 = sum(weights_p2)

prob_p1 = [
    weights_p1[0] / sum_p1,
    weights_p1[1] / sum_p1
]

prob_p2 = [
    weights_p2[0] / sum_p2,
    weights_p2[1] / sum_p2
]

print("Player 1 probabilities:", prob_p1)
print("Player 2 probabilities:", prob_p2)


# STEP 2: Compute expected payoff of each pure strategy

# Player 1
up_payoff = (
    payoff_p1[0][0] * prob_p2[0] + payoff_p1[0][1] * prob_p2[1]
)

down_payoff = (
    payoff_p1[1][0] * prob_p2[0] + payoff_p1[1][1] * prob_p2[1]
)

expected_p1 = [up_payoff, down_payoff]

print("Expected payoffs P1:", expected_p1)


# Player 2
left_payoff = (
    payoff_p2[0][0] * prob_p1[0] +
    payoff_p2[1][0] * prob_p1[1]
)

right_payoff = (
    payoff_p2[0][1] * prob_p1[0] +
    payoff_p2[1][1] * prob_p1[1]
)

expected_p2 = [left_payoff, right_payoff]

print("Expected payoffs P2:", expected_p2)


# STEP 3: Multiplicative weights update

weights_p1[0] = weights_p1[0] * math.exp(eta * expected_p1[0])
weights_p1[1] = weights_p1[1] * math.exp(eta * expected_p1[1])

weights_p2[0] = weights_p2[0] * math.exp(eta * expected_p2[0])
weights_p2[1] = weights_p2[1] * math.exp(eta * expected_p2[1])

print("Updated weights P1:", weights_p1)
print("Updated weights P2:", weights_p2)


# STEP 4: Normalize again to get new probabilities

sum_p1 = sum(weights_p1)
sum_p2 = sum(weights_p2)

prob_p1 = [
    weights_p1[0] / sum_p1,
    weights_p1[1] / sum_p1
]

prob_p2 = [
    weights_p2[0] / sum_p2,
    weights_p2[1] / sum_p2
]

print("New probabilities P1:", prob_p1)
print("New probabilities P2:", prob_p2)