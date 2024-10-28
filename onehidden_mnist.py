import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
from utils import OneHiddenLayer, pgd
import matplotlib.pyplot as plt

import fire, os, itertools

def mmap(x, digits=[0, 1]):
    if x == digits[0]:
        return -1
    elif x == digits[1]:
        return 1

def binarize(labels, digits=[0, 1]):
    return labels.apply_(lambda x: mmap(x, digits))

def ld_mnist(digits=[0, 1], train_size=None):
    """Load training and test data."""
    train_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    test_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )

    # Load MNIST dataset
    train_dataset = MNIST(root="./", transform=train_transforms, download=True)
    test_dataset = MNIST(root="./", train=False, transform=test_transforms, download=True)

    train_idx = torch.logical_or(train_dataset.targets == digits[0], train_dataset.targets == digits[1])
    train_dataset.data = train_dataset.data[train_idx]
    train_dataset.targets = binarize(train_dataset.targets[train_idx], digits)
    if train_size is not None:
        train_dataset.data = train_dataset.data[:train_size]
        train_dataset.targets = train_dataset.targets[:train_size]

    test_idx = torch.logical_or(test_dataset.targets == digits[0], test_dataset.targets == digits[1])
    test_dataset.data = test_dataset.data[test_idx]
    test_dataset.targets = binarize(test_dataset.targets[test_idx], digits)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_size, shuffle=True, num_workers=1
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=train_size, shuffle=False, num_workers=1
    )
    return train_loader, test_loader

def test_model(model, dataloader, device="cpu"):
    model.eval()
    accuracy, loss = 0, 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        data = data.view(-1, 28*28)
        data = pgd(model, data, target, eps=config['eps'], alpha=config['eps']/5, num_iter=10)
        with torch.no_grad():
            output = model(data).squeeze_()
            loss += torch.sum(torch.exp(- output * target))
            accuracy += torch.sum(torch.sign(output) == target).float()
    
    loss /= len(dataloader.dataset)
    accuracy /= len(dataloader.dataset)
    return loss, accuracy

def main():
    path = f'./results/mnist'
    for k, v in config.items():
        if ('sweep' in k) or (k == 'start_epoch'):
            continue
        if isinstance(v, list):
            v = '_'.join(map(str, v)) # e.g. [0, 1] -> '0_1'
        path = os.path.join(path, f'{k}_{v}')
    os.makedirs(path, exist_ok=True)

    LR, seed, EPOCHS = 1e-4, config['seed'], 4000
    if config["width"] > 2000 and config["algorithm"] == 'CD':
        EPOCHS = 2500

    if config['algorithm'] == 'CD':
        # LR = 3e-2
        # LR = 4e-3
        LR = 4e-2

    # Set the seed for reproducibility (of init of network)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    train_loader, test_loader = ld_mnist(digits=config['digits'], train_size=config['train_size'])

    model = OneHiddenLayer(hidden_dim=config['width'], scale=config['scale'])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    if device == "cuda":
        model = model.cuda()
    if config['algorithm'] == 'GD':
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    elif config['algorithm'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())

    if config['start_epoch'] > 0:
        model.load_state_dict(torch.load(os.path.join(path, f'model_{config["start_epoch"]}.pt')))
        log = torch.load(os.path.join(path, 'log.pt'))
    else:
        log = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'l2_margin': [],
            'linf_margin': [],
            'l1_margin': [],
            'smooth_l2_margin': [],
            'smooth_linf_margin': [],
            'smooth_l1_margin': []
        }

    count = 0
    for epoch in range(EPOCHS):
        # if epoch > 5 and log['train_acc'][-1] > 0.4 and count == 0:
        #     count += 1
        #     LR /= 5
        log['train_loss'].append(0)
        log['train_acc'].append(0)
        log['l2_margin'].append(np.inf)
        log['linf_margin'].append(np.inf)
        log['l1_margin'].append(np.inf)
        log['smooth_l2_margin'].append(0)
        log['smooth_linf_margin'].append(0)
        log['smooth_l1_margin'].append(0)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28*28)
            data = pgd(model, data, target, eps=config['eps'], alpha=config['eps']/5, num_iter=10)
            if config['algorithm'] == 'GD' or config['algorithm'] == 'Adam':    
                optimizer.zero_grad()
            else:
                # custom grad cleaning for coordinate/sign descent
                with torch.no_grad():
                    if model.linear1.weight.grad is not None:
                        model.linear1.weight.grad = None
                    if model.linear2.weight.grad is not None:
                        model.linear2.weight.grad = None

            output = model(data).squeeze_()
            loss = torch.sum(torch.exp(- output * target))
            
            log['train_loss'][-1] += loss.item()
            log['train_acc'][-1] += torch.sum(torch.sign(output) == target).float().cpu()
            with torch.no_grad():
                log['l2_margin'][-1] = min(log['l2_margin'][-1], model.normalized_margin(data, target, 2).item())
                log['linf_margin'][-1] = min(log['linf_margin'][-1], model.normalized_margin(data, target, 1).item())
                log['l1_margin'][-1] = min(log['l1_margin'][-1], model.normalized_margin(data, target, float('inf')).item())
            loss.backward()
            if config['algorithm'] == 'GD' or config['algorithm'] == 'Adam':
                optimizer.step()
            elif config['algorithm'] == 'CD':
                with torch.no_grad():            
                    grads = torch.cat([p.grad.view(-1) if p.grad is not None else torch.zeros_like(p.view(-1)) for p in model.parameters()])
                    params = torch.cat([p.view(-1) for p in model.parameters()])
                    largest_grad, largest_grad_index = torch.max(torch.abs(grads), dim=0)
                    params[largest_grad_index] -= LR * grads[largest_grad_index]
                    a, b = torch.split(params, [len(model.linear1.weight.flatten()), len(model.linear2.weight.flatten())])
                    model.linear1.weight.data = a.view(model.linear1.weight.shape)
                    model.linear2.weight.data = b.view(model.linear2.weight.shape)
            elif config['algorithm'] == 'SD': # sign gradient descent
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p -= LR * torch.sign(p.grad)

        with torch.no_grad():
            parameters = torch.cat([p.view(-1) for p in model.parameters()])
            l2 = torch.norm(parameters, 2)**2
            linf = torch.norm(parameters, float('inf'))**2
            l1 = torch.norm(parameters, 1)**2
            log['smooth_l2_margin'][-1] = - np.log(log['train_loss'][-1]) / l2
            log['smooth_linf_margin'][-1] = - np.log(log['train_loss'][-1]) / l1
            log['smooth_l1_margin'][-1] = - np.log(log['train_loss'][-1]) / linf
        log['train_loss'][-1] /= len(train_loader.dataset)
        log['train_acc'][-1] /= len(train_loader.dataset)
        test_loss, test_acc = test_model(model, test_loader, device)
        log['test_loss'].append(test_loss.item())
        log['test_acc'].append(test_acc.item())

        print(f"Epoch {epoch} train loss: {log['train_loss'][-1]:.3f}")
        print(f"Epoch {epoch} train acc: {log['train_acc'][-1]:.3f}")
        print(f"Epoch {epoch} test loss: {log['test_loss'][-1]:.3f}")
        print(f"Epoch {epoch} test acc: {log['test_acc'][-1]:.3f}")
        print(f"Epoch {epoch} l2 margin: {log['l2_margin'][-1]:.5f}")
        print(f"Epoch {epoch} linf margin: {log['linf_margin'][-1]:.5f}")
        print(f"Epoch {epoch} l1 margin: {log['l1_margin'][-1]:.5f}")
        print(f"Epoch {epoch} smooth l2 margin: {log['smooth_l2_margin'][-1]:.5f}")
        print(f"Epoch {epoch} smooth linf margin: {log['smooth_linf_margin'][-1]:.5f}")
        print(f"Epoch {epoch} smooth l1 margin: {log['smooth_l1_margin'][-1]:.5f}")
        print('---------------.-------------------')

    torch.save(log, os.path.join(path, 'log.pt'))
    torch.save(model.state_dict(), os.path.join(path, f'model_{config["start_epoch"] + EPOCHS}.pt'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(log['train_loss'], label='Train')
    ax1.plot(log['test_loss'], label='Test')
    ax1.set_title(f'(Robust) Loss {config["algorithm"]}')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(log['train_acc'], label='Train')
    ax2.plot(log['test_acc'], label='Test')
    ax2.set_title(f'(Robust) Accuracy {config["algorithm"]}')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'log.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(log['l2_margin'])
    ax1.set_title(f'L2 margin {config["algorithm"]}')
    ax1.set_xlabel('Epoch')

    ax2.plot(log['linf_margin'])
    ax2.set_title(f'Linf margin {config["algorithm"]}')
    ax2.set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'margins.pdf'), dpi=300, bbox_inches='tight')
    plt.show()


def retrieve_config(sweep_step):
    grid = {
        'digits': [[2, 7]],
        'train_size': [100, 250],
        'eps': [0, 0.2, 0.3],
        'width': [784],
        'algorithm': ['GD', 'SD'],
        'seed': [0, 1, 2],
        'scale': [0.01],
        'start_epoch': [8000]
    }

    grid_setups = list(
        dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())
    )
    step_grid = grid_setups[sweep_step - 1]  # slurm var will start from 1

    config = {
        'sweep_step': sweep_step,
        'digits': step_grid['digits'],
        'train_size': step_grid['train_size'],
        'eps': step_grid['eps'],
        'width': step_grid['width'],
        'algorithm': step_grid['algorithm'],
        'seed': step_grid['seed'],
        'scale': step_grid['scale'],
        'start_epoch': step_grid['start_epoch']
    }

    return config

def pre_main(sweep_step):
    global config
    config = retrieve_config(sweep_step)
    print(config)
    print('---------.--------')
    main()

if __name__ == '__main__':
    fire.Fire(pre_main)