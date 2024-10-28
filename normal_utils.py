import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class Linear(torch.nn.Module):
    def __init__(self, in_dim=28*28, out_dim=1):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.linear(x)

class DiagonalNet(nn.Module):
    def __init__(self, in_dim, init_scale, L):
        super().__init__()
        self.u = nn.Parameter(init_scale / ((in_dim * 2) ** 0.5) * torch.ones(in_dim))
        self.v = nn.Parameter(init_scale / ((in_dim * 2) ** 0.5) * torch.ones(in_dim))
        self.L = L
    
    def get_w(self):
        return self.u ** self.L - self.v ** self.L
    
    def forward(self, x):
        return (x @ self.get_w()).unsqueeze(-1)

def geometric_margin(X, y, w, p=2, bias=False):
    if bias:
        w_star = w / torch.linalg.norm(w[:-1], ord=p)
    else:
        try:
            w_star = w / torch.linalg.norm(w, ord=p)
        except NotImplementedError: # MPS
            w_star = w / torch.linalg.norm(w.to('cpu'), ord=p)
    margin = torch.min(y * X @ w_star.to(X.device))
    return margin

def average_geometric_margin(X, y, w, p=2, bias=False):
    if bias:
        w_star = w / torch.linalg.norm(w[:-1], ord=p)
    else:
        try:
            w_star = w / torch.linalg.norm(w, ord=p)
        except NotImplementedError: # MPS
            w_star = w / torch.linalg.norm(w.to('cpu'), ord=p)
    # calculate the average margin over all the data points
    margin = torch.mean(y * X @ w_star.to(X.device))
    # make sure that the margin is a scalar
    assert margin.shape == ()
    return margin

def prediction_margin(X, y, w, p_star=1, eps=0):
    pred_margin = torch.min(y * X @ w) - eps * torch.linalg.norm(w, ord=p_star)
    return pred_margin

def adversarial_margin(X, y, w, p=2, eps=0):
    pred_margin = torch.min(y * X @ w) - eps * torch.linalg.norm(w, ord=1)
    norm_w = torch.linalg.norm(w, ord=p)
    return pred_margin / norm_w

def accuracy(model, x, y):
    with torch.no_grad():
        y_hat = model(x)
        return torch.mean((torch.sign(y_hat) == y).float())

def robust_accuracy(model, w, x, y, eps):
    x_tilde = x - eps * y @ torch.sign(w)
    y_hat = model(x_tilde)
    return torch.mean((torch.sign(y_hat) == y).float())
    
def generate_data(n_samples, dim, seed=42, w_star=None, rho=0, teacher_sparsity=None, data_sparsity=None, precision=0):
    np.random.seed(seed)
    if w_star is None:
        if teacher_sparsity is None:
            w_star = np.random.normal(size=(dim, 1))
        else:
            assert teacher_sparsity <= dim
            values = [-1., precision, 1.]
            prob = dim / teacher_sparsity
            p = [1 / (2*prob), 1 - 1/prob, 1 / (2*prob)]
            w_star = np.random.choice(values, size=(dim, 1), p=p)

            # if w_star is all zeros, resample
            while np.count_nonzero(w_star) == 0:
                w_star = np.random.choice(values, size=(dim, 1), p=p)
            print("sparse w_star is", w_star.T)
            print("w_star has {} non-zero entries".format(np.count_nonzero(w_star)))
    if data_sparsity is None:
        X = np.random.normal(size=(n_samples, dim))
    else:
        assert data_sparsity <= dim
        values = [-1., precision, 1.]
        prob = dim / data_sparsity
        p = [1 / (2*prob), 1 - 1/prob, 1 / (2*prob)]
        X = np.random.choice(values, size=(n_samples, dim), p=p)
        
        # resample rows with all zeros
        zero_rows = np.where(~X.any(axis=1))[0]
        while len(zero_rows) > 0:
            for row in zero_rows:
                X[row] = np.random.choice(values, size=(dim,), p=p)
            zero_rows = np.where(~X.any(axis=1))[0]
    
    y = 2.*(X @ w_star > 0).astype(int) - 1
    X = X + rho * y * np.linalg.norm(w_star, 1) * w_star.T / np.linalg.norm(w_star, 2)**2

    
    return X, y, w_star

def generalization_experiment(D_train, D_test, algorithm='GD', lr=1e-2, epochs=10001, eps=0.3, seed=42, alpha=np.inf, verbose=False, LOSS_THR=1e-3, model_type='linear', loss_type='exp'):
    if algorithm not in ['GD', 'CD']:
        raise NotImplementedError
    
    # for initialization (?)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "mps" if torch.backends.mps.is_built() else "cpu"
    X_train, y_train = torch.from_numpy(D_train[0]).float().to(device=device), torch.from_numpy(D_train[1]).float().to(device=device)
    X_test, y_test = torch.from_numpy(D_test[0]).float().to(device=device), torch.from_numpy(D_test[1]).float().to(device=device)
    n_samples, dim = X_train.shape

    # Define model
    if model_type == 'linear':
        model = Linear(dim, 1)
    elif model_type == 'diagonal':
        model = DiagonalNet(dim, 1e-3, 2)
    if device == "cuda":
        model = model.cuda()
    elif device == "mps":
        model = model.to(device="mps")

    # init model with zero
    if model_type == 'linear':
        with torch.no_grad():
            model.linear.weight.zero_()
    
    # set learning rate, and optimizer (if GD) - 'proof' strategy is the convex upper bound on the learning rate (see our Theorem 1)
    if model_type == 'diagonal':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif algorithm == 'GD':
        if (lr=='best'):
            vals = torch.linalg.svdvals(X_train.to(device='cpu'))
            lr = 1 / vals[0]**2
        if (lr == 'proof'):
            # Adapted from the proof of Theorem 1 in https://arxiv.org/pdf/2006.05987.pdf (lol - autogenerated by Copilot - citation not true)
            x_2 = torch.linalg.norm(X_train, ord=np.inf, dim=1)
            assert x_2.shape == (n_samples,)
            x_max = torch.max(x_2)
            with torch.no_grad():
                if alpha == np.inf:
                    r_star_norm = torch.norm(model.linear.weight, 1)
                else:
                    r_star_norm = torch.sum(torch.log(1 + torch.exp(alpha * model.linear.weight)) + torch.log(1 + torch.exp(- alpha * model.linear.weight))) / alpha
                init_loss = torch.sum(torch.exp(- model(X_train) * y_train + eps*r_star_norm))
                lr = 2**(1/2) / ((x_max+eps)**2 * init_loss)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif lr=='proof':
        x_infs = torch.linalg.norm(X_train, ord=np.inf, dim=1)
        assert x_infs.shape == (n_samples,)
        x_max = torch.max(x_infs)
        with torch.no_grad():
            if alpha == np.inf:
                r_star_norm = torch.norm(model.linear.weight, 1)
            else:
                r_star_norm = torch.sum(torch.log(1 + torch.exp(alpha * model.linear.weight)) + torch.log(1 + torch.exp(- alpha * model.linear.weight))) / alpha
            init_loss = torch.sum(torch.exp(- model(X_train) * y_train + eps*r_star_norm))
            _lr = 2**(1/2) / ((x_max+eps)**2 * init_loss)


    # Train model
    log = dict(train_loss=[], test_loss=[], l2_norm=[], l1_norm=[], margininf=[], margin2=[], adv_margininf=[], adv_margin2=[], avg_margin=[], trainacc=[], testacc=[], trainrobacc=[], testrobacc=[], return_val=None)
    prec, etapl = 1e-8, 1e5
    epoch = 0
    while True:
        epoch += 1
        train_loss = 0.

        if algorithm == 'GD':
            optimizer.zero_grad()
        else:
            # custom grad cleaning for coordinate descent
            with torch.no_grad():
                for p in model.parameters():
                    if (p.grad is not None):
                        p.grad = None
            
        y_hat = model(X_train)
        if loss_type == 'exp':
            loss = torch.sum(torch.exp(- y_hat * y_train))
        elif loss_type == 'correlation':
            loss = torch.sum(- y_hat * y_train)
        if loss.isnan():
            print('loss is nan')
            # need to adapt to return values in case success
            break

        # update
        loss.backward()
        if algorithm == 'GD':
            if (lr == 'proof'):
                _lr = 1 / ((x_max+eps)**2 * loss.item() + prec)
                _lr = min(_lr, etapl)
                optimizer.param_groups[0]['lr'] = _lr
            optimizer.step()
        else:
            with torch.no_grad():
                # coordinate descent
                p = model.linear.weight
                idx = torch.argmax(torch.abs(p.grad))
                unit = nn.functional.one_hot(idx, p.grad.shape[1]).float().to(device=p.grad.device)
                if (lr == 'proof'):
                    _lr = 1 / ((x_max+eps)**2 * loss.item() + prec)
                    _lr = min(_lr, etapl)
                    p -= _lr * p.grad[:, idx] * unit
                else:
                    p -= lr * p.grad[:, idx] * unit

        # logging
        train_loss += loss.item()
        log['train_loss'].append(train_loss)
        # printing
        if epoch % 5000 == 1 and verbose:
            with torch.no_grad():
                if alpha == np.inf:
                    if model_type == 'diagonal':
                        r_star_norm = torch.norm(model.get_w(), 1)
                    else:
                        r_star_norm = torch.norm(model.linear.weight, 1)
                else:
                    if model_type == 'diagonal':
                        r_star_norm = torch.sum(torch.log(1 + torch.exp(alpha * model.get_w())) + torch.log(1 + torch.exp(- alpha * model.get_w()))) / alpha
                    else:
                        r_star_norm = torch.sum(torch.log(1 + torch.exp(alpha * model.linear.weight)) + torch.log(1 + torch.exp(- alpha * model.linear.weight))) / alpha
                log['test_loss'].append(torch.sum(torch.exp(- model(X_test) * y_test + eps*r_star_norm)).item())
                if model_type == 'diagonal':
                    w = model.get_w().reshape(1, -1)
                else:
                    w = model.linear.weight
                log['l2_norm'].append(torch.norm(w, 2).cpu().detach().numpy())
                log['l1_norm'].append(torch.norm(w, 1).cpu().detach().numpy())
                adv_margin2 = adversarial_margin(X_train, y_train, w.T, p=2, eps=eps)
                log['adv_margin2'].append(adv_margin2.cpu().detach().numpy())
                adv_margininf = adversarial_margin(X_train, y_train, w.T, p=1, eps=eps)
                log['adv_margininf'].append(adv_margininf.cpu().detach().numpy())
                margin_l2 = geometric_margin(X_train, y_train, w.T, p=2)
                log['margin2'].append(margin_l2.cpu().detach().numpy())
                margin_linf = geometric_margin(X_train, y_train, w.T, p=1)
                log['margininf'].append(margin_linf.cpu().detach().numpy())
                avg_margin = average_geometric_margin(X_train, y_train, w.T, p=2)
                log['avg_margin'].append(avg_margin.cpu().detach().numpy())
                train_acc = accuracy(model, X_train, y_train)
                test_acc = accuracy(model, X_test, y_test)
                rob_train_acc = robust_accuracy(model, w, X_train, y_train, eps)
                rob_test_acc = robust_accuracy(model, w, X_test, y_test, eps)
                log['trainacc'].append(train_acc)
                log['testacc'].append(test_acc)
                log['trainrobacc'].append(rob_train_acc)
                log['testrobacc'].append(rob_test_acc)
            print(
                "epoch {}/, train loss: {:.3f}, train acc {:.3f}, test acc {:.3f}, train robust acc {:.3f}, test robust acc {:.3f}, l2 margin {:.3f}, linf margin {:.3f}, avg margin {:.3f}".format(
                epoch, train_loss, 
                train_acc, test_acc,
                rob_train_acc, rob_test_acc,
                margin_l2, margin_linf, avg_margin
                ))
            # if epoch > 10000 and epochs=='converge':
            #     print('True-l1 (prev) loss', _loss)
            if epoch > 10000 and prev  < train_loss:
                print('loss increased')
                return None
            prev = train_loss
        if epochs == 'converge':
            # compare based on true l1 penalized loss (for fair comparison between algorithms)
            with torch.no_grad():
                if model_type == 'diagonal':
                    r_star_norm = torch.norm(model.get_w(), 1)
                else:
                    r_star_norm = torch.norm(model.linear.weight, 1)
                _loss = torch.sum(torch.exp(- y_hat * y_train + eps*r_star_norm))
            if _loss < LOSS_THR or (epoch > 2e5):
                with torch.no_grad():
                    if alpha == np.inf:
                        if model_type == 'diagonal':
                            r_star_norm = torch.norm(model.get_w(), 1)
                        else:
                            r_star_norm = torch.norm(model.linear.weight, 1)
                    else:
                        if model_type == 'diagonal':
                            r_star_norm = torch.sum(torch.log(1 + torch.exp(alpha * model.get_w())) + torch.log(1 + torch.exp(- alpha * model.get_w()))) / alpha
                        else:
                            r_star_norm = torch.sum(torch.log(1 + torch.exp(alpha * model.linear.weight)) + torch.log(1 + torch.exp(- alpha * model.linear.weight))) / alpha
                    log['test_loss'].append(torch.sum(torch.exp(- model(X_test) * y_test + eps*r_star_norm)).item())
                    if model_type == 'diagonal':
                        w = model.get_w().reshape(1, -1)
                    else:
                        w = model.linear.weight
                    log['l2_norm'].append(torch.norm(w, 2).cpu().detach().numpy())
                    log['l1_norm'].append(torch.norm(w, 1).cpu().detach().numpy())
                    adv_margin2 = adversarial_margin(X_train, y_train, w.T, p=2, eps=eps)
                    log['adv_margin2'].append(adv_margin2.cpu().detach().numpy())
                    adv_margininf = adversarial_margin(X_train, y_train, w.T, p=1, eps=eps)
                    log['adv_margininf'].append(adv_margininf.cpu().detach().numpy())
                    margin_l2 = geometric_margin(X_train, y_train, w.T, p=2)
                    log['margin2'].append(margin_l2.cpu().detach().numpy())
                    margin_linf = geometric_margin(X_train, y_train, w.T, p=1)
                    log['margininf'].append(margin_linf.cpu().detach().numpy())
                    avg_margin = average_geometric_margin(X_train, y_train, w.T, p=2)
                    log['avg_margin'].append(avg_margin.cpu().detach().numpy())
                    train_acc = accuracy(model, X_train, y_train)
                    test_acc = accuracy(model, X_test, y_test)
                    rob_train_acc = robust_accuracy(model, w, X_train, y_train, eps)
                    rob_test_acc = robust_accuracy(model, w, X_test, y_test, eps)
                    log['trainacc'].append(train_acc)
                    log['testacc'].append(test_acc)
                    log['trainrobacc'].append(rob_train_acc)
                    log['testrobacc'].append(rob_test_acc)
                # log['return_val'] = loss.item()
                break
        # elif epoch >= epochs and log['margininf'][-1] > 0:
        elif epoch >= epochs:
            break
            
    if loss.isnan():
        return None
    log['epoch'] = epoch
    return log