import torch
import torch.nn as nn
from torch import optim
import csv
import time
import random
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(68, 50)
        self.output = nn.Linear(50, 6)
    
    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.output(x)
        x = torch.softmax(x, dim=0)
        return x

def getIndexMax(tensor):
    t_max = tensor.max()
    for i in range(len(tensor)):
        if tensor[i] == t_max:
            return i

def getIndexMin(tensor):
    t_min = tensor.min()
    for i in range(len(tensor)):
        if tensor[i] == t_min:
            return i

emotions = ["angry", "fear", "happy", "sad", "surprise", "neutral"]

# model = torch.load("model.pt")
model = Model()

lossFunc = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay= 1e-6, momentum = 0.9, nesterov = True)

trainingData = []
validData = []
trainloader = []
validloader = []
with open("trainingdata.csv", "r") as csv_read:
    csvRead = csv.DictReader(csv_read)
    for row in csvRead:
        trainingData.append([int(row["emotion"]), list(map(lambda x: x / 2304, list(map(int, row["predictionPoints"].split()))))])
    for data in trainingData:
        x = torch.Tensor(data[1])
        y = torch.Tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        y[data[0]] = 1.0
        trainloader.append([x, y])
with open("testingdata.csv", "r") as csv_read:
    csvRead = csv.DictReader(csv_read)
    for row in csvRead:
        validData.append([int(row["emotion"]), list(map(lambda x: x / 2304, list(map(int, row["predictionPoints"].split()))))])
    for data in trainingData:
        x = torch.Tensor(data[1])
        y = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        y[data[0]] = 1.0
        validloader.append([x, y])



for epoch in range(1, 100): ## run the model for 100 epochs
    train_loss, valid_loss = [], []

    model.train()
    for data in trainloader:
        optimizer.zero_grad()
        output = model(data[0])
        loss = lossFunc(torch.softmax(output, dim=0), data[1])
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    correct = 0
    model.eval()
    for data in validloader:
        
        output = model(data[0])
        if emotions[getIndexMax(output)] == emotions[getIndexMax(data[1])]:
            correct += 1

        loss = lossFunc(torch.sigmoid(output), data[1])
        valid_loss.append(loss.item())
    if (epoch % 20 == 0):
        torch.save(model, "model.pt")
    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss), "Accuracy: ", correct / 1077 * 100)

