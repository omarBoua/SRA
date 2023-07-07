import numpy as np
output = [5.3375e-05, 6.1625e-05 , 5.7000e-05, 6.1375e-05, 5.9e-05, 5.6750e-05 ,5.6750e-05  ]
print(np.mean(output))
print("cov", np.std(output) / np.mean(output))
print(np.mean([40,42,36,39,36,41,38]))