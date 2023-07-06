import numpy as np
from imblearn.over_sampling import SMOTE

class BalancedPopulationGenerator:
    def __init__(self, nMC):
        self.nMC = nMC
        self.S = None
        self.classes = None

    def performance_function(self, c1, c2, m, r, t1, F1):
        w0 = np.sqrt((c1 * c2)/m)
        return 3 * r - np.abs(2 * F1 * np.sin(w0*t1/2)/ (m*w0**2))


    def generate_data(self):
        m = np.random.normal(1, 0.05, size=self.nMC)
        c1 = np.random.normal(1, 0.1, size=self.nMC)
        c2 = np.random.normal(0.1, 0.01, size=self.nMC)
        r = np.random.normal(0.5, 0.05, size=self.nMC)
        F1 = np.random.normal(1, 0.2, size=self.nMC)
        t1 = np.random.normal(1, 0.2, size=self.nMC) 
        self.classes = np.zeros(self.nMC)

        self.S = np.column_stack((c1, c2, m, r, t1, F1))
        for i in range(self.nMC):
            self.classes[i] = 0 if (self.performance_function(self.S[i, 0], self.S[i, 1], self.S[i,2], self.S[i,3], self.S[i,4], self.S[i,5]) >=0) else 1  # Evaluate performance function

    def balance_population(self):
        smote = SMOTE(k_neighbors = 2, sampling_strategy='auto')
        self.S, self.classes = smote.fit_resample(self.S, self.classes)
        print(len(self.S))
        if len(self.S) > 2 * self.nMC:
            self.S = self.S[:2 * self.nMC]
            self.classes = self.classes[:2*self.nMC]
        elif len(self.S) < 2 * self.nMC:
            num_samples_to_add = 2 * self.nMC - len(self.S)
            m = np.random.normal(1, 0.05, size=num_samples_to_add)
            c1 = np.random.normal(1, 0.1, size=num_samples_to_add)
            c2 = np.random.normal(0.1, 0.01, size=num_samples_to_add)
            r = np.random.normal(0.5, 0.05, size=num_samples_to_add)
            F1 = np.random.normal(1, 0.2, size=num_samples_to_add)
            t1 = np.random.normal(1, 0.2, size=num_samples_to_add) 

            additional_samples = np.column_stack((c1, c2, m, r, t1, F1))

            for i in range(num_samples_to_add):
                if(self.performance_function(additional_samples[i,0]
                                             ,additional_samples[i,1]
                                             ,additional_samples[i,2]
                                             , additional_samples[i,3]
                                             , additional_samples[i,4]
                                             , additional_samples[i,5])>= 0):
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
        print("Total classes:", len(self.classes))
    def get_population(self):
        #smote = SMOTE(k_neighbors = 2, sampling_strategy='auto')

        return (self.S,self.classes)


