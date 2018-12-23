from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import util

def corrupt_image(image_data):
    corruption_factor = np.random.random_sample()
    corrupted_image = image_data + corruption_factor * np.random.normal(loc=0.0, scale=1.0, size=image_data.shape)
    corrupted_image = np.clip(corrupted_image, 0., 1.)
    
    return corrupted_image

class MNIST(Dataset):
    def __init__(self):
        self.X , self.Y = util.get_mnist()
    
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        sample = { 'X' : self.X[idx], 'X_corrupted' : corrupt_image(self.X[idx])}
        return sample

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.layer1 = nn.Linear(784, 300)
        self.layer2 = nn.Linear(300, 784)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        return x

dataset = MNIST()
dl = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
criterion = nn.MSELoss()
model = NeuralNet()
model.double()
optimizer = optim.RMSprop(model.parameters(), lr=0.01)
print("Started training :")

for i in range(30):
    #batch = batch_iterator.next()
    #print(batch)
    for batch in dl:
        #print(batch['X'].shape)
        optimizer.zero_grad()   # zero the gradient buffers
        output = model(batch['X_corrupted'])
        loss = criterion(output, batch['X'])
        #print(loss)
        loss.backward()
        optimizer.step()
    print("Loss after epoch {0} : {1}".format(i+1,loss))
print("Training ended.")

i = np.random.choice(len(dataset))
x = dataset[i]['X']
corrupt_x = corrupt_image(x)
im = model(torch.DoubleTensor(corrupt_x)).detach().numpy().reshape(28, 28)
print(type(im))
plt.subplot(1,3,1)
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.title("Original")
plt.subplot(1,3,2)
plt.imshow(corrupt_x.reshape(28,28), cmap='gray')
plt.title("Noised image")
plt.subplot(1,3,3)
plt.imshow(im, cmap='gray')
plt.title("Denoised image")
plt.show()
