import numpy as np
import torch
import torch.nn as nn
import math

class Linear(torch.nn.Module):
    def __init__(self, in_dim=28*28, out_dim=1):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.linear(x)

class SparseFirstLayer(torch.nn.Module):
    def __init__(self, in_dim=100, hidden_dim=100, out_dim=1, fraction=0.5):
        super(SparseFirstLayer, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=False)

        num_elements = int(in_dim * (1-fraction))
        for i in range(hidden_dim):
            indices = torch.randperm(in_dim)[:num_elements]
            self.linear1.weight.data[i, indices] = 0.0

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        return self.linear2(x)

    def normalized_margin(self, x, y, p=2):
        parameters = torch.cat([p.view(-1) for p in self.parameters()])
        fx = self.forward(x) / torch.norm(parameters, p)**2
        return torch.min(y * fx.flatten())

class SparseOneHiddenLayer(torch.nn.Module):
    def __init__(self, in_dim=100, hidden_dim=100, out_dim=1, fraction=0.5):
        super(SparseOneHiddenLayer, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=False)

        num_elements = int(self.linear1.weight.data.numel() * (1-fraction))
        indices = torch.randperm(self.linear1.weight.data.numel())[:num_elements]
        self.linear1.weight.data.view(-1)[indices] = 0.0

        num_elements = int(self.linear2.weight.data.numel() * (1-fraction))
        indices = torch.randperm(self.linear2.weight.data.numel())[:num_elements]
        self.linear2.weight.data.view(-1)[indices] = 0.0

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        return self.linear2(x)

    def normalized_margin(self, x, y, p=2):
        parameters = torch.cat([p.view(-1) for p in self.parameters()])
        fx = self.forward(x) / torch.norm(parameters, p)**2
        return torch.min(y * fx.flatten())

class OneHiddenLayer(torch.nn.Module):
    # homogeneous network
    def __init__(self, in_dim=28*28, out_dim=1, hidden_dim=100, scale=1.0, lambda1=1.0):
        super(OneHiddenLayer, self).__init__()
        self.init_scale = scale
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=False)

        stdv = 1. / math.sqrt(lambda1*in_dim)
        nn.init.uniform_(self.linear1.weight.data, -scale*stdv, scale*stdv)
        stdv = 1. / math.sqrt(hidden_dim)
        nn.init.uniform_(self.linear2.weight.data, -scale*stdv, scale*stdv)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        return self.linear2(x)

    def normalized_margin(self, x, y, p=2):
        parameters = torch.cat([p.view(-1) for p in self.parameters()])
        fx = self.forward(x) / torch.norm(parameters, p)**2
        return torch.min(y * fx.flatten())

class TwoHiddenLayer(torch.nn.Module):
    def __init__(self, in_dim=28*28, out_dim=1, hidden_dim=100, scale=1.0, lambda1=1.0):
        super(TwoHiddenLayer, self).__init__()
        self.init_scale = scale
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear3 = nn.Linear(hidden_dim, out_dim, bias=False)

        stdv = 1. / math.sqrt(lambda1*in_dim)
        nn.init.uniform_(self.linear1.weight.data, -scale*stdv, scale*stdv)
        stdv = 1. / math.sqrt(hidden_dim)
        nn.init.uniform_(self.linear2.weight.data, -scale*stdv, scale*stdv)
        stdv = 1. / math.sqrt(hidden_dim)
        nn.init.uniform_(self.linear3.weight.data, -scale*stdv, scale*stdv)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        return self.linear3(x)

    def normalized_margin(self, x, y, p=2):
        parameters = torch.cat([p.view(-1) for p in self.parameters()])
        fx = self.forward(x) / torch.norm(parameters, p)**3
        return torch.min(y * fx.flatten())

class HomogeneousDeep(torch.nn.Module):
    def __init__(self, in_dim=28*28, out_dim=10, hidden_dim=100, depth=2, scale=1.0):
        super(HomogeneousDeep, self).__init__()
        self.init_scale = scale
        self.depth = depth
        self.linears = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False)])
        for _ in range(depth - 1):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.output = nn.Linear(hidden_dim, out_dim, bias=False)

        for i in range(depth - 1):
            stdv = 1. / math.sqrt(hidden_dim)
            nn.init.uniform_(self.linears[i].weight.data, -scale*stdv, scale*stdv)
        stdv = 1. / math.sqrt(hidden_dim)
        nn.init.uniform_(self.output.weight.data, -scale*stdv, scale*stdv)

    def forward(self, x):
        for i in range(self.depth):
            x = self.linears[i](x)
            x = torch.relu(x)
        return self.output(x)

class CNN(nn.Module):
    def __init__(self, out_dim=10, width=1024, scale=1.0):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 64, width)
        self.fc2 = nn.Linear(width, out_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 7 * 7 * 64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def geometric_margin(X, y, w, p=2, bias=False):
    if bias:
        w_star = w / torch.linalg.norm(w[:-1], ord=p)
    else:
        w_star = w / torch.linalg.norm(w, ord=p)
    margin = torch.min(y * X @ w_star)
    return margin

def accuracy(model, x, y):
    with torch.no_grad():
        y_hat = model(x)
        return torch.mean((torch.sign(y_hat) == y).float())

def robust_accuracy(model, x, y, eps):
    with torch.no_grad():
        for p in model.parameters():
            w = p
            break
        x_tilde = x - eps * y @ torch.sign(w)
        y_hat = model(x_tilde)
        return torch.mean((torch.sign(y_hat) == y).float())

def pgd(model, x, y, eps, alpha=1e-2, num_iter=10):
    """Construct PGD adversarial examples on the examples x"""
    delta = torch.zeros_like(x, requires_grad=True)
    # delta = torch.randn_like(x, requires_grad=True)
    for _ in range(num_iter):
        output = model(x+delta).squeeze_()
        loss = torch.sum(torch.exp(- output * y))
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-eps, eps)
        delta.data = torch.min(torch.max(delta.data, x - eps), x + eps)
        delta.grad.zero_()
    return torch.clamp(x+delta.detach(), 0, 1)

def pgd_multiclass(model, x, y, eps, alpha=1e-2, num_iter=10):
    """Construct PGD adversarial examples on the examples x"""
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(num_iter):
        output = model(x+delta)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-eps, eps)
        delta.data = torch.min(torch.max(delta.data, x - eps), x + eps)
        delta.grad.zero_()
    return torch.clamp(x+delta.detach(), 0, 1)