import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import warnings
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import SS_surrogate as ss
import math
from scipy.stats import norm

warnings.filterwarnings("ignore")

def gcal1(X,mu,sdev): #lower probability example
    global function_calls
    function_calls +=1
    return 0.5 * (X[0]-2)**2 - 1.5 *(X[1]-5)**3 - 3 

def gcal2(X,mu,sdev):   #four branches example
    global function_calls 
    function_calls += 1
    k = 6
    term1 = 3 + 0.1 * (X[0] - X[1])**2 - (X[0] + X[1])/(np.sqrt(2))
    term2 = 3 + 0.1 * (X[0] - X[1])**2 + (X[0] + X[1])/(np.sqrt(2))
    term3 = (X[0] - X[1]) + k / (2**0.5)
    term4 = (X[1] - X[0]) + k / (2**0.5)
    return min(term1, term2, term3, term4) 

def gcal3(X,mu,sdev): #dynamic response example
    global function_calls 
    function_calls += 1
    X = mu + sdev * X
    w0 = np.sqrt((X[0] * X[1])/X[2])
    k = 3
    temp = k * X[3] - np.abs(2 * X[5] * np.sin(w0*X[4]/2)/ (X[2]*w0**2))
    return -temp
def eff(g_hat_values, sigma_g_values):
    a = 0
    epsilon = 2 * np.square(sigma_g_values)
    term1 = (g_hat_values - a) * (2 * norm.cdf((a - g_hat_values) / sigma_g_values) - norm.cdf((a - epsilon - g_hat_values) / sigma_g_values) - norm.cdf((a + epsilon - g_hat_values) / sigma_g_values))
    term2 = -sigma_g_values * (2 * norm.pdf((a - g_hat_values) / sigma_g_values) - norm.pdf((a - epsilon - g_hat_values) / sigma_g_values) - norm.pdf((a + epsilon - g_hat_values) / sigma_g_values))
    term3 = norm.cdf((a + epsilon - g_hat_values) / sigma_g_values) - norm.cdf((a - epsilon - g_hat_values) / sigma_g_values)
    eff_values = term1 + term2 + term3
    return eff_values


# Stage 1: Generation of Monte Carlo population
nMC = 70000
dim = 2
S= np.random.normal(0,1,size = (nMC,dim))
mu = np.array([1,0.1,1,0.5,1,1])  # mean of rvs
sdev = np.array([.1,.01,.05,.05,.2,.2])  # sdev of rvs
function_calls = 0





# Stage 2: Definition of initial design of experiments (DoE)
N1 = 12

n_EDini = N1 
selected_indices = np.random.choice(len(S), N1, replace=False)

DoE = S[selected_indices]


Pf_values = np.zeros(N1)  # Array to store performance function evaluations
for i in range(N1):
    Pf_values[i] = gcal1(DoE[i],mu ,sdev)  # Evaluate performance function


# Stage 3: Computation of Kriging model
kernel = C(1.0, (1e-3, 1e3)) * RBF(np.repeat([0.5], dim), (1e-3, 1e2))  # Decreased lower bound from 1e-2 to 1e-3
kriging = GaussianProcessRegressor()
kriging.fit(DoE, Pf_values)
iter =0
function_calls_values = []
pf_hat_values = []
Nlevel = 500
p_0 = 0.1
while True:
    # Stage 4: Prediction by Kriging and estimation of probability of failure
    nMC = len(S)
    G_hat, kriging_std = kriging.predict(S,return_std=True)
    
    #estimate probability of failure using subset simulation
    Pf_hat,cov_ss,gamma = ss.SS(kriging,dim=dim, Nlevel= Nlevel)
    
    # Stage 5: Identification of the best next point to evaluate
    learning_values = eff(G_hat, kriging_std)
    x_best_index = np.argmax(learning_values)
    x_best = S[x_best_index]
    # Stage 6: Stopping condition on learning
    stopping_condition = max(learning_values) <= 0.001  
   
 
    print(max(learning_values))
    function_calls_values.append(function_calls)
    pf_hat_values.append(Pf_hat)
    # Stage 7: Update of the previous design of experiments with the best point
    if stopping_condition:
        # Stopping condition met, learning is stopped

        cov_pf = np.sqrt(1 - Pf_hat) / (np.sqrt(Pf_hat* nMC) )
        cov_threshold = 0.05
        if cov_ss < 0.1:
           
            # Coefficient of variation is acceptable, stop AK-MCS
            print("AK-MCS finished. Probability of failure: {:.4e}".format(Pf_hat))
            print("Coefficient of variation: {:.4%}".format(cov_ss))
            print("Number of calls to the performance function", function_calls)
            break
            # Stage 10: End of AK-MCS
        else:
            r = 3
            m = np.log(Pf_hat)/np.log(p_0)
            print("gamma", gamma)
            print("m", m)
            
            Nlevel = 2 * Nlevel#math.ceil((np.abs(np.log(Pf_hat))**r * (1 + gamma) * (1-p_0)/ (p_0 * np.abs(np.log(p_0))**r * cov_threshold**2)) / m)
            print("nlevel", Nlevel)
            # Coefficient of variation is too high, update population
            
         
            # Go back to Stage 4
    else:
        # Stopping condition not met, update design of experiments
        x_best_performance = gcal1(x_best,mu,sdev)
        Pf_values = np.concatenate((Pf_values, [x_best_performance]))
        DoE = np.vstack((DoE, x_best))
        kriging.fit(DoE, Pf_values)
        # Go back to Stage 4
    iter += 1
    
    print("iter ",iter, ": ",Pf_hat)

# Plotting pf_hat values vs. function_calls
plt.plot(function_calls_values, pf_hat_values, 'b-')
plt.xlabel('Function calls')
plt.ylabel('Probability of failure')

last_point_calls = function_calls_values[-1]
last_point_pf_hat = pf_hat_values[-1]
plt.plot(last_point_calls, last_point_pf_hat, 'ro')


# Display the number of iterations
plt.text(0.95, 0.95, f'Iterations until convergence: {iter}',
         verticalalignment='top', horizontalalignment='right',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

plt.show()  