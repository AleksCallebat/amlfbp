import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import torch.optim as optim
import time
import copy
import torch.onnx
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("running on device",device)


### METADATA ###
import numpy as np
with open("cleaning.txt","rb") as file:
    data=file.read()
lines=data.decode("utf8").replace("\r","").split("\n")
metalabels_100=[line.split("\t")[0] for line in lines]  
labels_100=[k for line in lines for k in line.split("\t")[1].split(",") ]

image_size=32
### DATA LOADING ### 
transform = transforms.Compose( 
    [
    transforms.RandomResizedCrop(image_size),
  #  transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    ])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
    shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

dataloaders={"train" : trainloader, "val" : testloader}

# functions to show an image
import numpy as np


dataiter = iter(trainloader)
images, labels = dataiter.next()
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes_100 = tuple(labels_100)

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__=="__main__" :
    net = Net()
    net.to(device)
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(4):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            
    print('Finished Training')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    print("exporting the model")
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    torch.onnx.export(net,dummy_input,"CIFAR_model.onnx",input_names=["data"],output_names=["classLabel"])
    print(dummy_input,len(dummy_input[0][0]))
