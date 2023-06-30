import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from data_generator import DataGenerator

def performance_function( x1, x2):
        k = 6
        term1 = 3 + 0.1 * (x1 - x2)**2 - (x1 + x2)/(np.sqrt(2))
        term2 = 3 + 0.1 * (x1 - x2)**2 + (x1 + x2)/(np.sqrt(2))
        term3 = (x1 - x2) + k / (2**0.5)
        term4 = (x2 - x1) + k / (2**0.5)
        
        return min(term1, term2, term3, term4)
pf_values = []
for i in range(10):
    # Stage 1: Generation of Monte Carlo population
    nMC =10000

    data_gen= DataGenerator()
    S = data_gen.generate_data(nMC)
    S = np.array(S)
    function_calls = 0


    # Stage 2: Definition of initial design of experiments (DoE)


    classes = np.zeros(nMC)  # Array to store performance function evaluations
    for i in range(nMC):
        classes[i] = 0 if (performance_function(S[i, 0], S[i, 1]) <=0) else 1  # Evaluate performance function
        function_calls += 1

    print(np.sum(classes <= 0))
    # Stage 3: Computation of MLP model
    mlp = MLPClassifier(hidden_layer_sizes=(100,100), activation= 'logistic', solver = 'adam',learning_rate = 'constant',max_iter= 100000, learning_rate_init= 0.01)  # Customize the hidden layer sizes as needed
    mlp.fit(S, classes)


    test_size =100000
    test_u1 = np.random.normal(0, 1, size=test_size)
    test_u2 = np.random.normal(0, 1, size=test_size)



    # Stage 4: prediction
    S_test = np.column_stack((test_u1, test_u2))
    #S_test = scaler.fit_transform(S_test)
    
    classes_hat = mlp.predict(S_test)
    print(classes_hat)
    

    Pf_hat = np.sum(classes_hat == 0) / test_size
    pf_values.append(Pf_hat)
    print(Pf_hat)
print(np.mean(pf_values))