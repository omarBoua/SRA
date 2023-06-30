import numpy as np

class DataGenerator:
    def __init__(self):
        self.countA = 0
        self.countB = 0
        self.S = []
        self.iter = 0
    
    def g(self, x1, x2):
        k = 6
        term1 = 3 + 0.1 * (x1 - x2)**2 - (x1 + x2) / np.sqrt(2)
        term2 = 3 + 0.1 * (x1 - x2)**2 + (x1 + x2) / np.sqrt(2)
        term3 = (x1 - x2) + k / np.sqrt(2)
        term4 = (x2 - x1) + k / np.sqrt(2)
        return min(term1, term2, term3, term4)
    
    def generate_data(self, num_samples):
        while len(self.S) < num_samples:
            x1 = np.random.normal(0, 1)
            x2 = np.random.normal(0, 1)

            if self.g(x1, x2) <= 0:
                if self.countA < num_samples // 2:
                    self.S.append((x1, x2))
                    self.countA += 1
            else:
                if self.countB < num_samples // 2:
                    self.S.append((x1, x2))
                    self.countB += 1

            self.iter += 1
        return self.S
