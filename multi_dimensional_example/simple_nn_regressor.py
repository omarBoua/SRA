import numpy as np
from sklearn.neural_network import MLPRegressor
#limit state function with two inputs x1 and x2
def g(X):
    n = len(X)
    return n + 3 * 0.2 * np.sqrt(n) - np.sum(X)


pf_values = []
function_calls = 0
for j in range(40):
    # Stage 1: Generation of Monte Carlo population
    nMC = 500
    n = 100 


    mu_lognormal = np.log(1/np.sqrt(0.2**2+1))

    sigma_lognormal = np.sqrt(np.log(1 + 0.2**2))

    S = np.random.lognormal(mean= mu_lognormal , sigma=sigma_lognormal, size=(nMC, n))
        

    # Stage 2: Definition of initial design of experiments (DoE)


    labels = np.zeros(nMC)  # Array to store performance function evaluations
    for i in range(nMC):
        labels[i] =  g(S[i])  # Evaluate performance function
        #labels[i] = np.tanh(labels[i])

    # Stage 3: Computation of MLP model

    mlp = MLPRegressor(hidden_layer_sizes=(10,10,10,10,10), activation='tanh',solver = 'lbfgs',  max_iter = 10000, verbose= True,learning_rate= "adaptive")
    mlp.fit(S, labels)

    test_size =100000
    test_S  = np.random.lognormal(mean= mu_lognormal , sigma=sigma_lognormal, size=(test_size, n))
    # Stage 4: prediction
   

    y_pred = mlp.predict(test_S)

    

    Pf_hat = np.sum(y_pred <= 0) / test_size
    print(Pf_hat)
    
    pf_values.append(Pf_hat)
    
print(np.mean(pf_values))
print(np.std(pf_values)/np.mean(pf_values))