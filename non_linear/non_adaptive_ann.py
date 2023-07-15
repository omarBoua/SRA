import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import math
#limit state function with two inputs x1 and x2
def LSF(x1,x2):
    return 10 - (x1**2 - 5 * math.cos(2*math.pi*x1)) - x2**2 - 5 * math.cos(2* math.pi * x2)

pf_values = []

#np.random.seed(29)
for i in range(10):
    # Stage 1: Generation of Monte Carlo population
    nMC = 2000
    x1 = np.random.normal(0, 1, size=nMC)
    x2 = np.random.normal(0, 1, size=nMC)
    S = np.column_stack((x1, x2))
    scaler = StandardScaler()
    S = scaler.fit_transform(S)


    # Stage 2: Definition of initial design of experiments (DoE)


    labels = np.zeros(nMC)  # Array to store performance function evaluations
    for i in range(nMC):
        labels[i] =  LSF(S[i, 0], S[i, 1])  # Evaluate performance function
        labels[i] = np.tanh(labels[i])

    # Stage 3: Computation of MLP model

    mlp = MLPRegressor(hidden_layer_sizes=(15,15,15,15,15), activation='tanh' ,solver = 'lbfgs',  max_iter = 10000)
    mlp.fit(S, labels)

    test_size =1000000
    test_u1 = np.random.normal(0, 1, size=test_size)
    test_u2 = np.random.normal(0, 1, size=test_size)



    # Stage 4: prediction
    S = np.column_stack((test_u1, test_u2))
    S = scaler.fit_transform(S)

    y_pred = mlp.predict(S)


    Pf_hat = np.sum(y_pred <= 0) / test_size
    pf_values.append(Pf_hat)
    
    
print(np.mean(pf_values))
print(np.std(pf_values) / np.mean(pf_values))