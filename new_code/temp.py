import numpy as np

action_list = ['fold', 'call', 'raise']
prob_list = [1, 0.3, 0.2]  # Probabilities must sum to 1

random_action = np.random.choice(action_list, p=prob_list)
print(random_action)