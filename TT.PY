import numpy as np

index = 5 


worst_model_indices = [2,3,5,35,6]
worst_model_indices = np.array(worst_model_indices)
k = np.where(worst_model_indices == index)[0][0]  # Get the index of the current model
print(k)