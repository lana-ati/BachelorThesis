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
weights_p1 = [1.0, 1.0]
weights_p2 = [1.0, 1.0]

# Learning rate
eta = 0.1

#weights to probabilities
sum_p1 = weights_p1[0] + weights_p1[1]
sum_p2 = weights_p2[0] + weights_p2[1]

p1_A = weights_p1[0] / sum_p1
p1_B = weights_p1[1] / sum_p1

p2_A = weights_p2[0] / sum_p2
p2_B = weights_p2[1] / sum_p2


#expected payoffs
u1 = (
    p1_A * p2_A * payoff_p1[0][0] + p1_A * p2_B * payoff_p1[0][1] +
    p1_B * p2_A * payoff_p1[1][0] + p1_B * p2_B * payoff_p1[1][1]
)

u2 = (
    p1_A * p2_A * payoff_p2[0][0] + p1_A * p2_B * payoff_p2[0][1] +
    p1_B * p2_A * payoff_p2[1][0] + p1_B * p2_B * payoff_p2[1][1]
)


#gradient directions
# Player 1: gain from switching to A vs B
u1_if_A = p2_A * payoff_p1[0][0] + p2_B * payoff_p1[0][1]
u1_if_B = p2_A * payoff_p1[1][0] + p2_B * payoff_p1[1][1]
grad_p1 = u1_if_A - u1_if_B

# Player 2: gain from switching to A vs B
u2_if_A = p1_A * payoff_p2[0][0] + p1_B * payoff_p2[0][1]
u2_if_B = p1_A * payoff_p2[1][0] + p1_B * payoff_p2[1][1]
grad_p2 = u2_if_A - u2_if_B


#gradient update
weights_p1[0] += eta * grad_p1
weights_p1[1] -= eta * grad_p1

weights_p2[0] += eta * grad_p2
weights_p2[1] -= eta * grad_p2


#projection
weights_p1[0] = max(weights_p1[0], 1e-8)
weights_p1[1] = max(weights_p1[1], 1e-8)

weights_p2[0] = max(weights_p2[0], 1e-8)
weights_p2[1] = max(weights_p2[1], 1e-8)

print("P1 strategy:", p1_A, p1_B)
print("P2 strategy:", p2_A, p2_B)
print("P1 payoff:", u1)
print("P2 payoff:", u2)
print("grad P1:", grad_p1)
print("grad P2:", grad_p2)
print("updated weights P1:", weights_p1)
print("updated weights P2:", weights_p2)