import numpy as np
from sklearn.neural_network import MLPRegressor
#limit state function with two inputs x1 and x2

def g(c1, c2, m, r, t1, F1):
    global function_calls
    

    w0 = np.sqrt((c1 * c2)/m)

    w0 = np.sqrt((c1 * c2)/m)
    function_calls += 1

    return 3 * r - np.abs(2 * F1 * np.sin(w0*t1/2)/ (m*w0**2))

pf_values = []

function_calls = 0
for j in range(25):
    #np.random.seed(1350)
    # Stage 1: Generation of Monte Carlo population
    nMC = 200
    m = np.random.normal(1, 0.05, size=nMC)
    c1 = np.random.normal(1, 0.1, size=nMC)
    c2 = np.random.normal(0.1, 0.01, size=nMC)
    r = np.random.normal(0.5, 0.05, size=nMC)
    F1 = np.random.normal(1, 0.2, size=nMC)
    t1 = np.random.normal(1, 0.2, size=nMC) 

    S = np.column_stack((c1, c2, m, r, t1, F1))




    # Stage 2: Definition of initial design of experiments (DoE)


    labels = np.zeros(nMC)  # Array to store performance function evaluations
    for i in range(nMC):
        labels[i] =  g(S[i, 0], S[i, 1], S[i,2], S[i,3], S[i,4], S[i,5])  # Evaluate performance function
        labels[i] = np.tanh(labels[i])

    # Stage 3: Computation of MLP model

    mlp = MLPRegressor(hidden_layer_sizes=(13,13), activation='tanh',solver = 'lbfgs',  max_iter = 100000)
    mlp.fit(S, labels)

    test_size =100000
    test_m = np.random.normal(1, 0.05, size=test_size)
    test_c1 = np.random.normal(1, 0.1, size=test_size)
    test_c2 = np.random.normal(0.1, 0.01, size=test_size)
    test_r = np.random.normal(0.5, 0.05, size=test_size)
    test_F1 = np.random.normal(1, 0.2, size=test_size)
    test_t1 = np.random.normal(1, 0.2, size=test_size) 


    # Stage 4: prediction
    S = np.column_stack((test_c1, test_c2, test_m, test_r, test_t1, test_F1))

    y_pred = mlp.predict(S)

    

    Pf_hat = np.sum(y_pred >= 0) / test_size
    print(Pf_hat)
    
    pf_values.append(Pf_hat)
    
print()
print(np.mean(pf_values))
print(np.std(pf_values)/np.mean(pf_values))