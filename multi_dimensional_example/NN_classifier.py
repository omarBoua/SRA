import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from BalancedPopulationGenerator import BalancedPopulationGenerator

def g(X):
    n = len(X)
    return n + 3 * 0.2 * np.sqrt(n) - np.sum(X)

pf_values = []
#np.random.seed(1222)

for i in range(30):
    # Stage 1: Generation of Monte Carlo population
    nMC = 10000
    n = 100 #40
    generator = BalancedPopulationGenerator(nMC, n)
    generator.generate_data()
    generator.balance_population()
    generator.print_population_info()
# Get the generated population

    S,classes = generator.get_population()
   
    # Stage 3: Computation of MLP model
    
    mlp = MLPClassifier(hidden_layer_sizes=(80,80), activation= 'logistic', solver = 'adam',max_iter= 100000)  # Customize the hidden layer sizes as needed
    mlp.fit(S, classes)


    test_size =100000
   
    mu_lognormal = np.log(1/np.sqrt(0.2**2+1))

    sigma_lognormal = np.sqrt(np.log(1 + 0.2**2))

    S_test = np.random.lognormal(mean= mu_lognormal , sigma=sigma_lognormal, size=(test_size, n))
       
    probabilities = mlp.predict_proba(S_test)
    print(probabilities)
    threshold = 0.2
    classes_hat = np.where(probabilities[:, 1] > threshold, 1, 0)

    Pf_hat = np.sum(classes_hat == 0) / test_size
    pf_values.append(Pf_hat)
    print(Pf_hat)
print(np.mean(pf_values))
print(np.std(pf_values))