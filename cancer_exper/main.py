import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pickle
import matplotlib.pyplot as plt

from algs.moments_accountant import compute_z
from main_utils import federated_average, compute_gradients, selective_federated_average, client_noise_sort_batch, \
    create_client_data
from utils.privacy_params import get_epsilons_batchsizes
from utils.aggregate import find_delta_model, models_to_matrix, R_pca
from config import get_parser

from opacus import PrivacyEngine
import tqdm
import os
from main_utils import federated_average, compute_gradients, selective_federated_average, client_noise_sort_batch, create_client_data



# 定义全局模型
class GlobalModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GlobalModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# 定义训练函数（带差分隐私）
def train_local_model_with_dp(model, data, epochs, lr, batch_size, max_grad_norm, nsc):
        # 保存初始参数用于梯度计算
        initial_params = {k: v.clone() for k, v in model.state_dict().items()}

        model.train()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(data[0], data[1])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 创建PrivacyEngine并包装优化器
        privacy_engine = PrivacyEngine()

        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=nsc,
            max_grad_norm=max_grad_norm,
        )

        for epoch in range(epochs):
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        # 获取隐私消耗
        # epsilon = privacy_engine.get_epsilon(delta=target_delta)
        # print(f"本地训练隐私消耗: (ε = {epsilon:.2f}, δ = {target_delta})")

        # 提取原始模型参数（去掉Opacus的包装）
        final_params = {}
        for name, param in model.named_parameters():
            if '_module.' in name:
                # 去掉Opacus添加的前缀
                original_name = name.replace('_module.', '')
                final_params[original_name] = param.data.clone()
        # 计算梯度（参数变化）
        gradients = compute_gradients(initial_params, final_params)
        return final_params, gradients


# 定义训练函数
def train_local_model(model, data, epochs, lr, batch_size):
    initial_params = {k: v.clone() for k, v in model.state_dict().items()}
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(data[0], data[1])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # 计算梯度（参数变化）
    gradients = compute_gradients(initial_params, model.state_dict())
    return model.state_dict(), gradients


def gaussian_kernel_weights(positions, center_idx, sigma=1.0, positions1=None, positions1_w=0.5):

    distances_sq = np.sum((positions - positions[center_idx]) ** 2, axis=1)
    if positions1 is not None:
        ## positions1 噪声程度
        min_val = positions1.min()
        max_val = positions1.max()
        if max_val - min_val == 0:
            positions1 = positions1 - min_val  # 避免除零，全部归一成0positions = positionselse:
        else:
            positions1 = (positions1 - min_val) / (max_val - min_val)
        distances_sq1 = np.sum((positions1 - positions1[center_idx]) ** 2, axis=1)
        distances_sq = distances_sq + distances_sq1 * positions1_w

    # 使用 sigma 计算高斯权重
    weights = np.exp(-distances_sq / (2 * args.sigma ** 2))

    # 归一化权重
    normalized_weights = weights / np.sum(weights)

    return normalized_weights


def main(use_ours, args, use_dp=True):

    # 加载数据
    cancers = load_breast_cancer()
    X = cancers.data
    Y = cancers.target

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 转换为PyTorch张量
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 初始化全局模型
    global_model = GlobalModel(input_dim, output_dim)
    epsilons_input, batches = get_epsilons_batchsizes(args.num_clients, 'Dist1')
    client_data = create_client_data(x_train_tensor, y_train_tensor, args.num_clients)
    method = args.method
    weights = np.array([len(client_data[i][1]) for i in range(len(client_data))])
    weights_aggregation_n = list(weights / np.sum(weights))
    deltas_input = np.array([0.0001] * args.num_clients)
    if method == 'epsilon_min':
        Z = [round(compute_z(epsilon=np.min(epsilons_input), dataset_size=weights[i], batch=batches[i], \
                             local_epochs=args.num_local_epochs, global_epochs=args.num_epochs, delta=deltas_input[i]), 2) \
             for i in range(len(epsilons_input))]
    elif method in ['Robust_HDP', 'PFA', 'DPFedAvg', 'WeiAvg', 'Ours', 'FedAvg']:
        Z = [round(compute_z(epsilon=epsilons_input[i], dataset_size=weights[i], batch=batches[i], \
                             local_epochs=args.num_local_epochs, global_epochs=args.num_epochs, delta=deltas_input[i]), 2) \
             for i in range(len(epsilons_input))]

    batches_epoch = [len(client_data[i][1]) for i in range(args.num_clients)]
    local_epochs_clients = [args.num_local_epochs] * args.num_clients
    client_sort_result = client_noise_sort_batch(weights, Z, batches, local_epochs_clients, batches_epoch,
                                                 kernel_size=5)

    if args.method == 'Robust_HDP':
        weights_agg_file = "weights_agg_{}_{}_{}.pkl".format(args.method, args.privacy_dist, args.seed)
        weights_agg_file = os.path.join('./', weights_agg_file)
    test_accs = []
    # 联邦学习训练过程
    for global_epoch in tqdm.tqdm(range(args.num_epochs)):
        print(f"\n全局训练轮次 [{global_epoch + 1}/{args.num_epochs}]")

        client_updates = []
        client_gradients_updates = []

        for client_id, (client_x, client_y) in enumerate(client_data):
            # 每个客户端从当前全局模型开始训练
            local_model = GlobalModel(input_dim, output_dim)
            local_model.load_state_dict(global_model.state_dict())
            if use_dp:
                # 带差分隐私
                local_state_dict, gradients = train_local_model_with_dp(
                    local_model,
                    (client_x, client_y),
                    args.num_local_epochs,
                    args.learning_rate,
                    batches[client_id],
                    max_grad_norm,
                    Z[client_id]
                )
            else:
                local_state_dict, gradients = train_local_model(
                    local_model,
                    (client_x, client_y),
                    args.num_local_epochs,
                    args.learning_rate,
                    batches[client_id]
                )
            client_updates.append(local_state_dict)
            client_gradients_updates.append(gradients)

        if use_ours:
            similarity_matrix, client_aggregated_params = selective_federated_average(global_model, client_updates,
                                                                                      client_gradients_updates,
                                                                                      args.cosine_threshold, client_sort_result, args.sigma)

        if args.method == 'Robust_HDP':
            if global_epoch == 0:

                delta_models = []
                for m in range(len(client_updates)):
                    local_mod = GlobalModel(input_dim, output_dim)
                    local_mod.load_state_dict(client_updates[m])
                    old_mod = copy.deepcopy(global_model)
                    delta_models.append(find_delta_model(local_mod, old_mod))
                delta_matrix_main = models_to_matrix(delta_models)

                delta_matrix = delta_matrix_main
                RPCA = R_pca(delta_matrix)
                L, S = RPCA.fit(tol=5e-8)

                S_col_norms_squared_inversed = np.array(
                    [1 / (np.linalg.norm(S[:, i]) ** 2) for i in range(S.shape[1])])

                weights_aggregation_noise = list(S_col_norms_squared_inversed / np.sum(S_col_norms_squared_inversed))
                print('The weights obtained from RPCA are: {}'.format(weights_aggregation_noise))
                weights_aggregation = [weights_aggregation_noise[i] for i in range(args.num_clients)]
                with open(weights_agg_file, 'wb') as f_out:
                    pickle.dump(weights_aggregation, f_out)

                del delta_models
                del delta_matrix
                del L
                del S

            else:
                if os.path.exists(weights_agg_file):
                    with open(weights_agg_file, 'rb') as f_in:
                        weights_aggregation = pickle.load(f_in)
                else:
                    raise ValueError('previoulsy computed Robust_HDP aggregation weights are not found')

        #############################
        elif args.method == 'WeiAvg':
            weights_aggregation = list(epsilons_input / np.sum(epsilons_input))

        ###############################
        elif args.method == 'FedAvg':
            weights_aggregation = weights_aggregation_n  # i.e. aggregating the uploaded noisy updates with "weights_aggregation_n"

        elif args.method == 'DPFedAvg' or args.method == 'epsilon_min':
            weights_aggregation = weights_aggregation_n

        if not use_dp:
            weights_aggregation = weights_aggregation_n
        if use_ours:
            averaged_state_dict = federated_average(global_model, client_aggregated_params, weights_aggregation)
        else:
            averaged_state_dict = federated_average(global_model, client_updates, weights_aggregation)
        # 更新全局模型
        global_model.load_state_dict(averaged_state_dict)

        # 在测试集上评估全局模型
        global_model.eval()
        with torch.no_grad():
            test_outputs = global_model(x_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test_tensor).float().mean()
            test_accs.append(accuracy)
    return test_accs

# 超参数
input_dim = 30
output_dim = 2

# 差分隐私参数
target_epsilon = 1.0
target_delta = 1e-5
max_grad_norm = 2.0

def single_run(seed):
    # 设置随机种子以确保可重复性
    torch.manual_seed(seed)
    np.random.seed(seed)
    ours_accs = main(True, args)
    o_c = [a.item() for a in ours_accs]
    accs = main(False, args)
    c = [a.item() for a in accs]

    start_accs = main(False, args, use_dp=False)
    s_c = [a.item() for a in start_accs]

    print(o_c)
    print(c)
    print(s_c)
    print(f'dis:{[float(o_c[i])-float(c[i]) for i in range(len(o_c))]}')
    steps = [step for step in range(1, len(ours_accs)+1)]

    # print(accs)
    # 画折线图
    plt.figure(figsize=(8, 6))
    plt.plot(steps, ours_accs, label='Ours', linewidth=2, color='red', linestyle='--', marker='o')
    plt.plot(steps, accs, label='Other', linewidth=2, color='blue', linestyle='--', marker='x')
    plt.plot(steps, start_accs, label='Original', linewidth=2, linestyle='--', marker='*')  # p

    # 图形美化
    plt.title(f"cosine_threshold={args.cosine_threshold}, sigma={args.sigma}", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    # plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    # single_run(seed=13)
    seeds = [21, 41, 13, 0, 63]


    def run_experiments(use_ours, use_dp=True):
        all_runs = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            accs = main(use_ours, args, use_dp=use_dp)
            accs = [a.item() for a in accs]
            all_runs.append(accs)
        return np.array(all_runs)  # shape = (len(seeds), global_epochs)


    # 三种方法
    ours_runs = run_experiments(True, use_dp=True)
    other_runs = run_experiments(False, use_dp=True)
    original_runs = run_experiments(False, use_dp=False)

    steps = np.arange(1, args.num_epochs + 1)

    # 每次实验单独的折线图
    for i in range(len(seeds)):
        plt.figure(figsize=(8, 6))
        plt.plot(steps, ours_runs[i], label='Ours', linewidth=2, color='red', linestyle='--', marker='o')
        plt.plot(steps, other_runs[i], label='Other', linewidth=2, color='blue', linestyle='--', marker='x')
        plt.plot(steps, original_runs[i], label='Original', linewidth=2, linestyle='--', marker='*')  # p
        # 图形美化
        plt.title(f"method={args.method} cosine_threshold={args.cosine_threshold}, sigma={args.sigma}", fontsize=14)
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        # plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    # 几次平均的折线图
    # 计算均值 & 标准差
    ours_mean, ours_std = ours_runs.mean(axis=0), ours_runs.std(axis=0)
    other_mean, other_std = other_runs.mean(axis=0), other_runs.std(axis=0)
    original_mean, original_std = original_runs.mean(axis=0), original_runs.std(axis=0)
    plt.figure(figsize=(8, 6))


    def plot_curve(x, mean, std, label, color, line_width=2, line_style='--'):
        plt.plot(x, mean, label=label, color=color, linewidth=line_width, linestyle=line_style)
        # plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        # plt.fill_between(x, mean - 0.5 * std, mean + 0.5 * std, color=color, alpha=0.2)
        # ci95 = 1.96 * std / np.sqrt(len(seeds))
        # plt.fill_between(x, mean - ci95, mean + ci95, color=color, alpha=0.2)
        sem = std / np.sqrt(len(seeds))
        plt.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)


    plot_curve(steps, ours_mean, ours_std, 'Ours', 'red', line_style='-')
    plot_curve(steps, other_mean, other_std, 'Other', 'blue', line_style='-')
    plot_curve(steps, original_mean, original_std, 'Original', 'green', line_style='-')

    plt.title(f"cosine_threshold={args.cosine_threshold}, sigma={sigma}", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Average Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
