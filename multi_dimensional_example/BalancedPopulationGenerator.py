import numpy as np
from imblearn.over_sampling import SMOTE

class BalancedPopulationGenerator:
    def __init__(self, nMC, n):
        self.nMC = nMC
        self.S = None
        self.n = n
        self.classes = None

    def performance_function(self, X):
        n = len(X)
        return n + 3 * 0.2 * np.sqrt(n) - np.sum(X)


    def generate_data(self):
         
        mu_lognormal = np.log(1/np.sqrt(0.2**2+1))

        sigma_lognormal = np.sqrt(np.log(1 + 0.2**2))

        self.S = np.random.lognormal(mean= mu_lognormal , sigma=sigma_lognormal, size=(self.nMC, self.n))



        self.classes = np.zeros(self.nMC)

        for i in range(self.nMC):
            self.classes[i] = 0 if (self.performance_function(self.S[i]) <= 0) else 1

    def balance_population(self):
        smote = SMOTE(k_neighbors = 2, sampling_strategy='auto')
        self.S, self.classes = smote.fit_resample(self.S, self.classes)

        if len(self.S) > 2 * self.nMC:
            self.S = self.S[:2 * self.nMC]
            self.classes = self.classes[:2*self.nMC]
        elif len(self.S) < 2 * self.nMC:
            num_samples_to_add = 2 * self.nMC - len(self.S)
            mu_lognormal = np.log(1/np.sqrt(0.2**2+1))

            sigma_lognormal = np.sqrt(np.log(1 + 0.2**2))

            additional_samples = np.random.lognormal(mean= mu_lognormal , sigma=sigma_lognormal, size=(num_samples_to_add, self.n))


           

            for i in range(num_samples_to_add):
                if(self.performance_function(additional_samples[i])<= 0):
                    self.classes = np.append(self.classes, np.array([0]))
                else:
                    self.classes = np.append(self.classes, np.array([1]))
            self.S = np.vstack((self.S, additional_samples))

    def print_population_info(self):
        classA = np.sum(self.classes == 0)
        classB = np.sum(self.classes == 1)
        print("classA:", classA)
        print("classB:", classB)
        print("Total samples:", len(self.S))
    
    def get_population(self):
        #smote = SMOTE(k_neighbors = 2, sampling_strategy='auto')

        return (self.S,self.classes)


