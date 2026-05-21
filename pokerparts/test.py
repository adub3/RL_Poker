import numpy as np

# Define action space
action_space = np.array([0, 1, 2, 3, 4], dtype=np.uint8)

# Save to binary file
np.save("action_space.npy", action_space)

# Load from binary file
loaded_action_space = np.load("action_space.npy")
print(loaded_action_space)

