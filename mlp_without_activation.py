import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random,time
import scipy.sparse as sp
import scipy.io as sio
from mlp import cifar_loaders
from mlp import to_onehot


class MLPNoactNet(nn.Module):
    def __init__(self):
        super(MLPNoactNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)  
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)  
        return x


if __name__ == "__main__":

    batch_size = 64
    test_batch_size = 64

    train_loader, _ = cifar_loaders(batch_size)
    _, test_loader = cifar_loaders(test_batch_size)
    
    Results = {"MLPnoact Epoch": [], "Train Loss": [], "Test Loss": [], "Accuracy": []}

    model = MLPNoactNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 10  
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = to_onehot(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / (i + 1)}')

        Results["MLPnoact Epoch"].append(epoch)
        Results["Train Loss"].append(running_loss)


    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            # labels = to_onehot(labels)
            loss += criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')

    Results["Accuracy"].append(accuracy)
    Results["Test Loss"].append(loss)

    sio.savemat('MLPnoact_res.mat', Results)