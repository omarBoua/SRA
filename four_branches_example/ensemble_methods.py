from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from statistics import mode
from BalancedPopulationGenerator import BalancedPopulationGenerator
import numpy as np
from sklearn.model_selection import train_test_split

def performance_function( x1, x2):
        k = 6
        term1 = 3 + 0.1 * (x1 - x2)**2 - (x1 + x2)/(np.sqrt(2))
        term2 = 3 + 0.1 * (x1 - x2)**2 + (x1 + x2)/(np.sqrt(2))
        term3 = (x1 - x2) + k / (2**0.5)
        term4 = (x2 - x1) + k / (2**0.5)
        
        return min(term1, term2, term3, term4)
pf_val = []
for _ in range(1):                   
    nMC = 25000
    generator = BalancedPopulationGenerator(nMC)
    generator.generate_data()
    generator.balance_population()
    generator.print_population_info()
    # Get the generated population
    S ,classes= generator.get_population()


    
    # Step 1: Split the dataset into training and validation/test sets

    # Assume you have X_train, y_train, X_val, y_val as your training and validation sets

    # Step 2: Create and train multiple MLP classifiers

    # Set hyperparameters for the MLP classifiers
    num_classifiers = 10  # Number of MLP classifiers in the ensemble
    hidden_layers = [(5,5),(100,), (20,20), (10,10), (5,), (50,50), (5,5,5),(50,), (100,100), (7,7,7)]  # Number of neurons in hidden layers


    classifiers = []
    for k in range(num_classifiers):
        # Create MLP classifier
        classifier = MLPClassifier(hidden_layer_sizes= (100,100), activation= 'logistic', solver = 'adam',max_iter= 100000, early_stopping = True)  # Customize the hidden layer sizes as needed

        
        # Train MLP classifier
        classifier.fit(S, classes)
        
        # Add trained classifier to the ensemble
        classifiers.append(classifier)

    # Step 4: Obtain predictions from each classifier
    test_size =1000000
    test_u1 = np.random.normal(0, 1, size=test_size)
    test_u2 = np.random.normal(0, 1, size=test_size)



    # Stage 4: prediction
    S_test = np.column_stack((test_u1, test_u2))


    predictions = []
    for classifier in classifiers:
        # Obtain predictions on the validation/test set
        classifier_proba = classifier.predict_proba(S_test)
        threshold = 0.05
        classifier_pred = np.where(classifier_proba[:, 1] > threshold, 1, 0)
        # Add predictions to the list
        predictions.append(classifier_pred)
        #print(np.sum(classifier_pred == 0) ) 
    # Step 5: Perform majority voting





    ensemble_predictions = []
    for i in range(len(S_test)):
        # Obtain the predictions for the i-th datapoint

        datapoint_predictions = [pred[i] for pred in predictions]
        
        # Perform majority voting to select the predicted class
        majority_vote = mode(datapoint_predictions)
        
        # Add the majority vote to the ensemble predictions
        ensemble_predictions.append(majority_vote)

    # Step 6: Evaluate the ensemble's performance

    #ensemble_accuracy = accuracy_score(y_val, ensemble_predictions)
    #print("Ensemble Accuracy:", ensemble_accuracy)

    # Note: You can further adjust and optimize the ensemble by experimenting with different hyperparameters, techniques, and evaluation metrics.
    ensemble_predictions = np.array(ensemble_predictions)
    pf_value = np.sum(ensemble_predictions == 0) / test_size
    pf_val.append(pf_value)
    print("pf: ", pf_value)
print("mean: ", np.mean(pf_val))
print("std: ", np.std(pf_val))