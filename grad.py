import numpy as np

# Matchups: 4 competitors, 3 matches
matchups = np.array([[0, 1],
                     [2, 3],
                     [1, 2]])

# Dummy gradients for each match
gradients = np.array([[0.5, -0.5],
                      [0.3, -0.3],
                      [0.2, -0.2]])

# Initialize the result array
num_competitors = 4
competitor_gradients = np.zeros(num_competitors, dtype=np.float32)

# Aggregate gradients


print(f'{competitor_gradients.shape=}')
print(f'{matchups[:,0].shape=}')
print(f'{gradients[:,0].shape=}')

np.add.at(competitor_gradients, matchups[:, 0], gradients[:, 0])
np.add.at(competitor_gradients, matchups[:, 1], gradients[:, 1])

print("Matchups:")
print(matchups)
print("\nGradients:")
print(gradients)
print("\nAggregated gradients per competitor:")
print(competitor_gradients)