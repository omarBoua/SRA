import numpy as np
losses = [1,5,9,2002,4,34,56,888]
num_layers_to_update = 3

worst_model_indices = np.argsort(losses)[-num_layers_to_update:]

print(worst_model_indices)
print(losses[worst_model_indices[0]])