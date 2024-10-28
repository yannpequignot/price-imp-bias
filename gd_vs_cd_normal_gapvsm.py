import fire, os, itertools
import matplotlib.pyplot as plt
import numpy as np
import torch

from normal_utils import Linear, geometric_margin, accuracy, robust_accuracy, generate_data, generalization_experiment

def main():
    PRECISION = 0
    #  Directory creation
    path = f'./results/normal'
    for k, v in config.items():
        if ('sweep' in k):
            continue
        path = os.path.join(path, f'{k}_{v}')
    os.makedirs(path, exist_ok=True)

    # gap vs m
    n_samples_list = [config['dim'] / 8, config['dim'] / 4, config['dim'] / 2, config['dim'], 2 * config['dim']]
    for dim in [config['dim']]:
        if dim > 1024:
            X_test, y_test, w_star = generate_data(n_samples=10*dim, dim=dim, seed=0, rho=config['distribution_margin'], teacher_sparsity=config['teacher_sparsity'], data_sparsity=config['data_sparsity'], precision=PRECISION)
        else:
            X_test, y_test, w_star = generate_data(n_samples=dim**2, dim=dim, seed=0, rho=config['distribution_margin'], teacher_sparsity=config['teacher_sparsity'], data_sparsity=config['data_sparsity'], precision=PRECISION)
        D_test = (X_test, y_test)
        gen_gap = {'CD': [], 'GD': []}
        clean_gap = {'CD': [], 'GD': []}
        l1_norm = {'CD': [], 'GD': []}
        l2_norm = {'CD': [], 'GD': []}
        adv_margin2 = {'CD': [], 'GD': []}
        adv_margininf = {'CD': [], 'GD': []}
        emp_gen_bound = {'CD': [], 'GD': []}
        for n_samples in n_samples_list:
            gen_gap['CD'].append([])
            gen_gap['GD'].append([])
            clean_gap['CD'].append([])
            clean_gap['GD'].append([])
            l1_norm['CD'].append([])
            l1_norm['GD'].append([])
            l2_norm['CD'].append([])
            l2_norm['GD'].append([])
            adv_margin2['CD'].append([])
            adv_margin2['GD'].append([])
            adv_margininf['CD'].append([])
            adv_margininf['GD'].append([])
            emp_gen_bound['CD'].append([])
            emp_gen_bound['GD'].append([])
            n_samples = int(n_samples)
            for seed in range(3):
                X_train, y_train, _ = generate_data(n_samples=n_samples, dim=dim, seed=seed, w_star=w_star, rho=config['distribution_margin'], teacher_sparsity=config['teacher_sparsity'], data_sparsity=config['data_sparsity'], precision=PRECISION)
                D_train = (X_train, y_train)

                print(f'---------- dim, n_samples {dim, n_samples} ---------')

                # get largest possible eps
                log = generalization_experiment(D_train, D_test, algorithm='CD', lr='proof', epochs=100001, eps=0., alpha=np.inf, verbose=True)
                max_eps = log['margininf'][-1].item()
                

                eps = max_eps * config['eps']
                print(f'---------- eps {eps} ---------')
                log_GD = generalization_experiment(D_train, D_test, algorithm='GD', lr='proof', epochs='converge', eps=eps, alpha=np.inf, verbose=True, LOSS_THR=1e-3)
                for alpha in [np.inf, 2., 1.]:
                    log_CD = generalization_experiment(D_train, D_test, algorithm='CD', lr='proof', epochs='converge', eps=eps, alpha=alpha, verbose=True, LOSS_THR=1e-3)
                    if log_CD is not None:
                        break
                    print('running with smaller alpha')
                gen_gap['CD'][-1].append(log_CD['trainrobacc'][-1].item() - log_CD['testrobacc'][-1].item())
                clean_gap['CD'][-1].append(log_CD['trainacc'][-1].item() - log_CD['testacc'][-1].item())
                gen_gap['GD'][-1].append(log_GD['trainrobacc'][-1].item() - log_GD['testrobacc'][-1].item())
                clean_gap['GD'][-1].append(log_GD['trainacc'][-1].item() - log_GD['testacc'][-1].item())
                l1_norm['CD'][-1].append(log_CD['l1_norm'][-1].item())
                l1_norm['GD'][-1].append(log_GD['l1_norm'][-1].item())
                l2_norm['CD'][-1].append(log_CD['l2_norm'][-1].item())
                l2_norm['GD'][-1].append(log_GD['l2_norm'][-1].item())
                adv_margin2['CD'][-1].append(log_CD['adv_margin2'][-1].item())
                adv_margin2['GD'][-1].append(log_GD['adv_margin2'][-1].item())
                adv_margininf['CD'][-1].append(log_CD['adv_margininf'][-1].item())
                adv_margininf['GD'][-1].append(log_GD['adv_margininf'][-1].item())
                data_norm_1 = np.linalg.norm(X_train, ord=np.inf, axis=1).max()
                emp_gen_bound['CD'][-1].append((data_norm_1 * np.sqrt(np.log2(dim)) + eps) / log_CD['adv_margininf'][-1].item())
                data_norm_2 = np.linalg.norm(X_train, ord=2, axis=1).max()
                emp_gen_bound['GD'][-1].append((data_norm_2 + eps * np.sqrt(dim)) / log_GD['adv_margin2'][-1].item())
                print(f'---------- CD, GD {gen_gap["CD"][-1], gen_gap["GD"][-1]} ---------')

    mean_CD = np.mean(gen_gap['CD'], axis=1)
    mean_GD = np.mean(gen_gap['GD'], axis=1)
    std_CD = np.std(gen_gap['CD'], axis=1)
    std_GD = np.std(gen_gap['GD'], axis=1)
    mean_clean_CD = np.mean(clean_gap['CD'], axis=1)
    mean_clean_GD = np.mean(clean_gap['GD'], axis=1)
    std_clean_CD = np.std(clean_gap['CD'], axis=1)
    std_clean_GD = np.std(clean_gap['GD'], axis=1)
    mean_l1_norm_CD = np.mean(l1_norm['CD'], axis=1)
    mean_l1_norm_GD = np.mean(l1_norm['GD'], axis=1)
    mean_l2_norm_CD = np.mean(l2_norm['CD'], axis=1)
    mean_l2_norm_GD = np.mean(l2_norm['GD'], axis=1)
    std_l1_norm_CD = np.std(l1_norm['CD'], axis=1)
    std_l1_norm_GD = np.std(l1_norm['GD'], axis=1)
    std_l2_norm_CD = np.std(l2_norm['CD'], axis=1)
    std_l2_norm_GD = np.std(l2_norm['GD'], axis=1)
    mean_adv_margin2_CD = np.mean(adv_margin2['CD'], axis=1)
    mean_adv_margin2_GD = np.mean(adv_margin2['GD'], axis=1)
    mean_adv_margininf_CD = np.mean(adv_margininf['CD'], axis=1)
    mean_adv_margininf_GD = np.mean(adv_margininf['GD'], axis=1)
    std_adv_margin2_CD = np.std(adv_margin2['CD'], axis=1)
    std_adv_margin2_GD = np.std(adv_margin2['GD'], axis=1)
    std_adv_margininf_CD = np.std(adv_margininf['CD'], axis=1)
    std_adv_margininf_GD = np.std(adv_margininf['GD'], axis=1)
    mean_emp_gen_bound_CD = np.mean(emp_gen_bound['CD'], axis=1)
    mean_emp_gen_bound_GD = np.mean(emp_gen_bound['GD'], axis=1)
    std_emp_gen_bound_CD = np.std(emp_gen_bound['CD'], axis=1)
    std_emp_gen_bound_GD = np.std(emp_gen_bound['GD'], axis=1)
    log = {'n_samples_list': n_samples_list, 'mean_CD': mean_CD, 'mean_GD': mean_GD, 'std_CD': std_CD,
            'std_GD': std_GD, 'mean_clean_CD': mean_clean_CD, 'mean_clean_GD': mean_clean_GD, 'std_clean_CD': std_clean_CD, 'std_clean_GD': std_clean_GD,
            'mean_l1_norm_CD': mean_l1_norm_CD, 'mean_l1_norm_GD': mean_l1_norm_GD, 'mean_l2_norm_CD': mean_l2_norm_CD, 'mean_l2_norm_GD': mean_l2_norm_GD,
            'std_l1_norm_CD': std_l1_norm_CD, 'std_l1_norm_GD': std_l1_norm_GD, 'std_l2_norm_CD': std_l2_norm_CD, 'std_l2_norm_GD': std_l2_norm_GD,
            'mean_adv_margin2_CD': mean_adv_margin2_CD, 'mean_adv_margin2_GD': mean_adv_margin2_GD, 'mean_adv_margininf_CD': mean_adv_margininf_CD, 'mean_adv_margininf_GD': mean_adv_margininf_GD,
            'std_adv_margin2_CD': std_adv_margin2_CD, 'std_adv_margin2_GD': std_adv_margin2_GD, 'std_adv_margininf_CD': std_adv_margininf_CD, 'std_adv_margininf_GD': std_adv_margininf_GD,
            'mean_emp_gen_bound_CD': mean_emp_gen_bound_CD, 'mean_emp_gen_bound_GD': mean_emp_gen_bound_GD, 'std_emp_gen_bound_CD': std_emp_gen_bound_CD, 'std_emp_gen_bound_GD': std_emp_gen_bound_GD}
    np.save(os.path.join(path, f'gapvsm_stats_{config["dim"]}_rho_{config["distribution_margin"]}_eps_{config["eps"]}.npy'), log)

def retrieve_config(sweep_step):
    grid = {
        'dim': [512],
        'teacher_sparsity': [23],
        'data_sparsity': [4],
        'distribution_margin': [0],
        'eps': [0., 0.25, 0.5]
    }

    grid_setups = list(
        dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())
    )
    step_grid = grid_setups[sweep_step - 1]  # slurm var will start from 1

    config = {
        'sweep_step': sweep_step,
        'dim': step_grid['dim'],
        'teacher_sparsity': step_grid['teacher_sparsity'],
        'data_sparsity': step_grid['data_sparsity'],
        'distribution_margin': step_grid['distribution_margin'],
        'eps': step_grid['eps']
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