from sklearn.datasets import make_regression
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RegressionData(Dataset):
    def __init__(self, num_samples=1000, num_features=1, num_targets=1, Noise=25.0, transform=None):
        self.X , self.Y = make_regression(n_samples=num_samples, n_features=num_features, noise=Noise)
        
        #Normalize the data across the dataset.
        X_min, X_max = np.min(self.X), np.max(self.X)
        self.X = np.array(list(map(lambda k: (k - X_min)/(X_max-X_min) , self.X) ))
        Y_min, Y_max = np.min(self.Y), np.max(self.Y)
        self.Y = np.array(list(map(lambda k: (k-Y_min)/(Y_max-Y_min), self.Y)))
        
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):

        sample = {'X': self.X[idx], 'Y':self.Y[idx]}

        if self.transform :
            sample = self.transform(sample)
        
        return sample

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()

        self.layer1 = nn.Linear(1,4)
        self.layer2 = nn.Linear(4,1)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        return x

dataset = RegressionData()
dl = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
batch_iterator = iter(dl)
criterion = nn.MSELoss()
model = NeuralNet()
model.double()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for i in range(50):
    #batch = batch_iterator.next()
    #print(batch)
    for batch in dl:
        optimizer.zero_grad()   # zero the gradient buffers
        output = model(batch['X'])
        loss = criterion(output, batch['Y'].unsqueeze(-1))
        #print(loss)
        loss.backward()
        optimizer.step()
    print(loss)

save(model.state_dict(), 'intro_to_pytorch')
