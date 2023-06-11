import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


def performance_function(x1,x2):
    return x2 - np.abs(np.tan(x1)) - 1
pf_values = []
for i in range(1):
    # Stage 1: Generation of Monte Carlo population
    nMC = 500
    x1 = np.random.normal(4, 2, size=nMC)
    x2 = np.random.normal(2.5, 2, size=nMC)
    S = np.column_stack((x1, x2))
    function_calls = 0


    # Stage 2: Definition of initial design of experiments (DoE)


    classes = np.zeros(nMC)  # Array to store performance function evaluations
    for i in range(nMC):
        classes[i] = 1 if (performance_function(S[i, 0], S[i, 1]) <0) else 0  # Evaluate performance function
        function_calls += 1


    # Stage 3: Computation of MLP model

    mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), activation= 'logistic', solver = 'adam',learning_rate = 'constant', learning_rate_init= 0.01)  # Customize the hidden layer sizes as needed
    mlp.fit(S, classes)


    test_size =100000
    test_u1 = np.random.normal(4, 2, size=test_size)
    test_u2 = np.random.normal(-2, 2, size=test_size)



    # Stage 4: prediction
    S = np.column_stack((test_u1, test_u2))

    classes_hat = mlp.predict(S)

    y_pred_class = np.where(classes_hat > 0.5, 1, 0)

    Pf_hat = np.sum(y_pred_class == 0) / test_size
    pf_values.append(Pf_hat)
    print(Pf_hat)
print(np.mean(pf_values))