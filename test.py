import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")


l = .50



def g(x1,x2):
    
    return 0.5 * (x1-2)**2 - 1.5 *(x2-5)**3 - 3 - (1-l) * 209.22541271


# Stage 1: Generation of Monte Carlo population
nMC = 700000   #for lower probabilities, set it to at least 1000000
x1 = np.random.normal(0, 1, size=nMC)
x2 = np.random.normal(0, 1, size=nMC)

S = np.column_stack((x1,x2))


G = np.zeros(nMC)  # Array to store performance function evaluations
for i in range(nMC):
    G[i] = g(S[i, 0], S[i, 1]) 
    
print("mean" , np.mean(G))

p_l = np.sum(G <= 0)/ nMC



from scipy.optimize import minimize

def func_to_minimize(params):
    a, b, c, q = params
    return (np.log(p_l) - np.log(q) + a * (l - b)**c)**2

initial_params = [0, 0.2, 0, 2] # Provide sensible initial guesses
result = minimize(func_to_minimize, initial_params, method='Newton-CG')

a, b, c, q = result.x
import matplotlib.pyplot as plt

print(a,b,c,q)
print(p_l)
print(q * np.exp(-a * (1-b)**c))

# Define a range of x values
x = np.linspace(0, 1, 100)  # generates 1000 points evenly spaced between 0 and 10
# Define the function
y = q * np.exp(-a * (x - b)**c)

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Plot of the function q * np.exp(-a * (x - b)**c)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

