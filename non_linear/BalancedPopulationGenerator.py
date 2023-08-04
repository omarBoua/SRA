import numpy as np
from imblearn.over_sampling import SMOTE
import math
class BalancedPopulationGenerator:
    def __init__(self, nMC):
        self.nMC = nMC
        self.S = None
        self.classes = None

    def performance_function(self, x1,x2):
        return 10 - (x1**2 - 5 * math.cos(2*math.pi*x1)) - x2**2 - 5 * math.cos(2* math.pi * x2)

    def generate_data(self):
        u1 = np.random.normal(0, 1, size=self.nMC)
        u2 = np.random.normal(0, 1, size=self.nMC)
        self.S = np.column_stack((u1, u2))
        self.classes = np.zeros(self.nMC)

        for i in range(self.nMC):
            self.classes[i] = 0 if (self.performance_function(self.S[i, 0], self.S[i, 1]) <= 0) else 1

    def balance_population(self):
        smote = SMOTE(k_neighbors = 2, sampling_strategy='auto')
        self.S, self.classes = smote.fit_resample(self.S, self.classes)

        if len(self.S) > 2 * self.nMC:
            self.S = self.S[:2 * self.nMC]
            self.classes = self.classes[:2*self.nMC]
        elif len(self.S) < 2 * self.nMC:
            num_samples_to_add = 2 * self.nMC - len(self.S)
            x1 = np.random.normal(0, 1, num_samples_to_add)
            x2 = np.random.normal(0, 1, num_samples_to_add)
            additional_samples = np.column_stack((x1, x2))

            for i in range(num_samples_to_add):
                if(self.performance_function(additional_samples[i,0],additional_samples[i,1])<= 0):
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


