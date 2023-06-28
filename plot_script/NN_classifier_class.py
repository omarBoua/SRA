import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def performance_function(x1, x2):
    return x2 - np.abs(np.tan(x1)) - 1

class PerformanceNet(nn.Module):
    def __init__(self):
        super(PerformanceNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10,1)
   

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


class PerformanceTrainer:
    def __init__(self, N_test = 100000, N1 = 500):
        self.model = PerformanceNet()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.pf_values = []
        self.N_test = N_test
        self.N1 = N1 
    def train_and_get_pf_values(self, num_iterations):
        for i in range(num_iterations):
            x1_train = np.random.normal(4, 2, size=self.N1)
            x2_train = np.random.normal(2.5, 2, size=self.N1)
            y_train = np.array(
                [1 if performance_function(x1, x2) < 0 else 0 for x1, x2 in zip(x1_train, x2_train)]
            )
            x_t = torch.tensor(np.column_stack((x1_train, x2_train)), dtype=torch.float32)
            y_train = y_train.reshape(-1, 1)
            y_t = torch.tensor(y_train, dtype=torch.float32)

            num_epochs = 1000
            for epoch in range(num_epochs):
                self.optimizer.zero_grad()
                outputs = self.model.forward(x_t)
                loss = self.criterion(outputs, y_t)
                loss.backward()
                self.optimizer.step()

            x1_test = np.random.normal(4, 2, size=self.N_test) #more samples
            x2_test = np.random.normal(-2, 2, size=self.N_test)

            x_test = torch.tensor(np.column_stack((x1_test, x2_test)), dtype=torch.float32)

            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Disable gradient calculation
                y_pred = self.model(x_test)


            y_pred = y_pred.numpy()



            threshold = 0.5
            y_pred_class = np.where(y_pred > threshold, 1, 0)
            Pf_hat = np.sum(y_pred_class == 0) / self.N_test


            
            
            self.pf_values.append(Pf_hat)
        return self.pf_values
    
