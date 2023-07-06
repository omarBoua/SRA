from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from BalancedPopulationGenerator import BalancedPopulationGenerator
import numpy as np


def performance_function( x1, x2):
        k = 6
        term1 = 3 + 0.1 * (x1 - x2)**2 - (x1 + x2)/(np.sqrt(2))
        term2 = 3 + 0.1 * (x1 - x2)**2 + (x1 + x2)/(np.sqrt(2))
        term3 = (x1 - x2) + k / (2**0.5)
        term4 = (x2 - x1) + k / (2**0.5)
        
        return min(term1, term2, term3, term4)
pf_values = []
for _ in range(25):
    nMC = 5000
    generator = BalancedPopulationGenerator(nMC )
    generator.generate_data()
    generator.balance_population()

    # Get the generated population
    S,classes = generator.get_population()



    # Create the base classifier
    base_classifier = MLPClassifier(hidden_layer_sizes=(25), activation= 'logistic', solver = 'adam',max_iter= 100000, early_stopping = True)  
    #mlp = MLPClassifier(hidden_layer_sizes=(5,5), activation= 'logistic', solver = 'adam',max_iter= 100000, early_stopping = True)  

    # Create the ensemble using bagging
    ensemble = BaggingClassifier(base_classifier, n_estimators=1)

    # Train the ensemble on the training set
    ensemble.fit(S, classes)

    # Evaluate the ensemble on the validation/test set
    test_size =1000000
    test_u1 = np.random.normal(0, 1, size=test_size)
    test_u2 = np.random.normal(0, 1, size=test_size)

    # Stage 4: prediction
    S_test = np.column_stack((test_u1, test_u2))

    proba = ensemble.predict_proba(S_test)
    print(proba)
    threshold = 0.05
    predictions = np.where(proba[:, 1] > threshold, 1, 0)
    
   
    #predictions = ensemble.predict(S_test)
    pf = np.sum(predictions == 0) / test_size
    pf_values.append(pf)
    print(pf)

print("mean: ", np.mean(pf_values))
print("std: ", np.std(pf_values))


