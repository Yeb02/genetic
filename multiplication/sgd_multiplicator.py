import numpy as np
from numpy import random
import torch
from torch import nn


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class NeuralNetwork(nn.Module):
    def __init__(self, sizes):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        modules = []
        nb_layers = len(sizes)
        for i in range(nb_layers - 2):
            modules.append(nn.Linear(sizes[i], sizes[i+1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(sizes[nb_layers-2], sizes[nb_layers-1]))
        self.linear_relu_stack = nn.Sequential(*modules)

        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



def test(model, loss_fn, nb_batches, samples_per_batch):
    model.eval()
    test_loss, mean_dev = 0, 0
    with torch.no_grad():
        for _ in range(nb_batches):

            x = torch.Tensor([[10*random.rand(), 10*random.rand()] for j in range(samples_per_batch)])
            y = torch.Tensor([[x[j][0]*x[j][1]] for j in range(samples_per_batch)])
            x, y = x.to(device), y.to(device)

            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            mean_dev += ((pred-y)**2).type(torch.float).sum().item()
            
    test_loss /= nb_batches
    mean_dev /= nb_batches*samples_per_batch
    return test_loss, mean_dev
    


def train(model, loss_fn, optimizer, nb_batches, samples_per_batch):
    model.train()
    for _ in range(nb_batches):
        x = torch.Tensor([[10*random.rand(), 10*random.rand()] for j in range(samples_per_batch)])
        y = torch.Tensor([[x[j][0]*x[j][1]] for j in range(samples_per_batch)])
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

training = True
calculus_test = False

if training:
    model = NeuralNetwork([2,10,1]).to(device)    
    epochs = 100
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-5)
    # w = torch.nn.parameter.Parameter(torch.Tensor([[0 for _ in range(2)] for _ in range(10)] ))
    # print(type(model.linear_relu_stack[0].weight))
    # print(model.linear_relu_stack[0].weight)
    # print(type(w))
    # print(w)
    # model.eval()
    # with torch.no_grad():
    #     model.linear_relu_stack[0].weight = torch.nn.parameter.Parameter(torch.Tensor([[0 for _ in range(2)] for _ in range(10)] ))

    # model.train()
    # print(model.linear_relu_stack[0].weight)

    for t in range(epochs):
        # print(model.linear_relu_stack[0].weight[0])
        train(model, loss_fn, optimizer, 1000, 10)
        test_loss, mean_dev = test(model, loss_fn, 1, 100)
        print(f"Epoch {t}: Mean of the average quadratic error on test data: {mean_dev:>0.1f}, Avg loss: {test_loss:>8f}")

    # print(model(torch.Tensor([[3., 1.]])), model(torch.Tensor([[9., 7.]])))
    torch.save(model.state_dict(), "multiplier_0_to_10.pth")


if calculus_test:
    model = NeuralNetwork([2,20,1])
    model.load_state_dict(torch.load("multiplier_0_to_10.pth"))

    x = input()
    while x != "q":
        a, b = x.split(" ")
        print(f"{float(a)} x {float(b)} = {model(torch.Tensor([[float(b), float(a)]]))[0][0]}.")
        x = input()