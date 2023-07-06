import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from BalancedPopulationGenerator import BalancedPopulationGenerator

def g(c1, c2, m, r, t1, F1):
    w0 = np.sqrt((c1 * c2)/m)
    return 3 * r - np.abs(2 * F1 * np.sin(w0*t1/2)/ (m*w0**2))

pf_values = []
#np.random.seed(1222)

for i in range(22):
    # Stage 1: Generation of Monte Carlo population
    nMC = 5000
    
    generator = BalancedPopulationGenerator(nMC)
    generator.generate_data()
    generator.balance_population()
    generator.print_population_info()
# Get the generated population

    S,classes = generator.get_population()
   
    # Stage 3: Computation of MLP model
    
    mlp = MLPClassifier(hidden_layer_sizes=(25), activation= 'logistic', solver = 'adam',learning_rate = 'constant',max_iter= 10000, learning_rate_init= 0.01)  # Customize the hidden layer sizes as needed
    mlp.fit(S, classes)


    test_size =100000
    test_m = np.random.normal(1, 0.05, size=test_size)
    test_c1 = np.random.normal(1, 0.1, size=test_size)
    test_c2 = np.random.normal(0.1, 0.01, size=test_size)
    test_r = np.random.normal(0.5, 0.05, size=test_size)
    test_F1 = np.random.normal(1, 0.2, size=test_size)
    test_t1 = np.random.normal(1, 0.2, size=test_size) 


    # Stage 4: prediction
    S = np.column_stack((test_c1, test_c2, test_m, test_r, test_t1, test_F1))

    classes_hat = mlp.predict(S)


    Pf_hat = np.sum(classes_hat == 0) / test_size
    pf_values.append(Pf_hat)
    print(Pf_hat)
print(np.mean(pf_values))