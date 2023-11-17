import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import scipy.io as sio
import random,time
from mlp import cifar_loaders


class CNN(nn.Module):
    def __init__(self, batchsize):
        super(CNN, self).__init__()
        self.batchsize = batchsize
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, 64)
        self.fc3 = nn.Linear(64, 10)  

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = x.view(-1, 128 * 4 * 4)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def train(model, train_loader, criterion, optimizer, epoch, results):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        target = torch.from_numpy(to_onehot(target))
        loss = criterion(output, target)
        epoch_loss += loss.data.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

    results["CNN Epoch"].append(epoch)
    results["Train Loss"].append(epoch_loss)
    return results


def test(model, test_loader, criterion, results):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).data.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    results["Test Loss"].append(test_loss)
    results["Accuracy"].append(accuracy)
    return results


def to_onehot(prelabel):
    k = len(np.unique(prelabel))

    label = np.zeros([prelabel.shape[0], k])
    label[range(prelabel.shape[0]), prelabel] = 1
    return label


if __name__ == "__main__":
    batch_size = 64
    test_batch_size = 64

    train_loader, _ = cifar_loaders(batch_size)
    _, test_loader = cifar_loaders(test_batch_size)

    Results = {"CNN Epoch": [], "Train Loss": [], "Test Loss": [], "Accuracy": []}

    model = CNN(batchsize=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)



    for epoch in range(10):  
        Results = train(model, train_loader, criterion, optimizer, epoch, Results)
        Results = test(model, test_loader, criterion, Results)

    sio.savemat('SNN_res.mat', Results)

