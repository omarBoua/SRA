import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from BalancedPopulationGenerator import BalancedPopulationGenerator

def performance_function( x1, x2):
        k = 6
        term1 = 3 + 0.1 * (x1 - x2)**2 - (x1 + x2)/(np.sqrt(2))
        term2 = 3 + 0.1 * (x1 - x2)**2 + (x1 + x2)/(np.sqrt(2))
        term3 = (x1 - x2) + k / (2**0.5)
        term4 = (x2 - x1) + k / (2**0.5)
        
        return min(term1, term2, term3, term4)
pf_values = []
proba_shiha = []
for i in range(25):
    # Stage 1: Generation of Monte Carlo population
    nMC =5000

    """  data_gen= DataGenerator()
    S = data_gen.generate_data(nMC)
    S = np.array(S)
    """
    """ 
    u1 = np.random.normal(0, 1, size=nMC)
    u2 = np.random.normal(0, 1, size=nMC)



    # Stage 4: prediction
    S = np.column_stack((u1, u2)) """ 


    function_calls = 0

    generator = BalancedPopulationGenerator(nMC)
    generator.generate_data()
    generator.balance_population()

# Get the generated population
    S,classes = generator.get_population()
    # Stage 2: Definition of initial design of experiments (DoE)
    """

    classes = np.zeros(nMC)  # Array to store performance function evaluations
      for i in range(nMC):
        classes[i] = 0 if (performance_function(S[i, 0], S[i, 1]) <=0) else 1  # Evaluate performance function
        function_calls += 1 """

    print(np.sum(classes <= 0))
    # Stage 3: Computation of MLP model
    mlp = MLPClassifier(hidden_layer_sizes=(25), activation= 'logistic', solver = 'adam',max_iter= 100000, early_stopping = False)  # Customize the hidden layer sizes as needed
    mlp.fit(S, classes)

    test_size =1000000
    test_u1 = np.random.normal(0, 1, size=test_size)
    test_u2 = np.random.normal(0, 1, size=test_size)



    # Stage 4: prediction
    S_test = np.column_stack((test_u1, test_u2))
    #S_test = scaler.fit_transform(S_test)
    
    probabilities = mlp.predict_proba(S_test)
    print(probabilities)
    threshold = 0.05 # increase to increase predicted probability 

# Convert probabilities to predicted class labels based on the threshold
    classes_hat = np.where(probabilities[:, 1] > threshold, 1, 0)

    true_classes = np.zeros(test_size)
    for i in range(test_size):
        if(performance_function(S_test[i,0], S_test[i,1]) <= 0):
            true_classes[i] = 0
        else: 
             true_classes[i] = 1
    tn, fp, fn, tp = confusion_matrix(true_classes, classes_hat).ravel()
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    
    print("TN: ", tn)
    print("FP: ", fp)
    print("FN: ", fn)
    print("TP: ", tp)
    print("true positive rate: ", tpr)
    print("false positive rate: ", fpr)
    proba_ = 1-np.count_nonzero(true_classes)/test_size

    print("shiha:", 1-np.count_nonzero(true_classes)/test_size)
    proba_shiha.append(proba_)
    Pf_hat = np.sum(classes_hat == 0) / test_size
    pf_values.append(Pf_hat)
    print(Pf_hat)
print(np.mean(pf_values))
print(np.std(pf_values)/np.mean(pf_values))
