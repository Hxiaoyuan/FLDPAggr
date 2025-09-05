import copy
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from opacus import PrivacyEngine
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from algs.moments_accountant import compute_z
from config import get_parser
from draw import aggregate_mean_roc, roc_from_scores, Ours_color, Others_color, Original_color
from main_utils import federated_average, compute_gradients, selective_federated_average, client_noise_sort_batch, \
    create_client_data
from utils.aggregate import find_delta_model, models_to_matrix, R_pca
from utils.privacy_params import get_epsilons_batchsizes


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


# 训练一次（返回指标序列 + 最终ROC用的 y_true/y_score）
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
                             local_epochs=args.num_local_epochs, global_epochs=args.num_epochs, delta=deltas_input[i]),
                   2) \
             for i in range(len(epsilons_input))]
    elif method in ['Robust_HDP', 'PFA', 'DPFedAvg', 'WeiAvg', 'Ours', 'FedAvg']:
        Z = [round(compute_z(epsilon=epsilons_input[i], dataset_size=weights[i], batch=batches[i], \
                             local_epochs=args.num_local_epochs, global_epochs=args.num_epochs, delta=deltas_input[i]),
                   2) \
             for i in range(len(epsilons_input))]

    batches_epoch = [len(client_data[i][1]) for i in range(args.num_clients)]
    local_epochs_clients = [args.num_local_epochs] * args.num_clients
    client_sort_result = client_noise_sort_batch(weights, Z, batches, local_epochs_clients, batches_epoch,
                                                 kernel_size=5)

    if args.method == 'Robust_HDP':
        weights_agg_file = "weights_agg_{}_{}_{}.pkl".format(args.method, args.privacy_dist, args.seed)
        weights_agg_file = os.path.join('./', weights_agg_file)

    test_accs, test_precisions, test_recalls, test_f1s = [], [], [], []
    y_true_final = None  # 用于ROC（最后一轮）
    y_score_final = None  # 用于ROC（最后一轮），为类别1的概率
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
                                                                                      args.cosine_threshold,
                                                                                      client_sort_result, args.sigma)

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
            probs = torch.softmax(test_outputs, dim=1)  # (N, 2)
            y_prob1 = probs[:, 1].cpu().numpy()  # 类别1的概率（二分类）
            _, predicted = torch.max(test_outputs, 1)

            accuracy = (predicted == y_test_tensor).float().mean().item()
            precision = precision_score(y_test_tensor.numpy(), predicted.numpy(), average="macro")
            recall = recall_score(y_test_tensor.numpy(), predicted.numpy(), average="macro")
            f1 = f1_score(y_test_tensor.numpy(), predicted.numpy(), average="macro")

            test_accs.append(accuracy)
            test_precisions.append(precision)
            test_recalls.append(recall)
            test_f1s.append(f1)

            # 仅保存最后一轮用于 ROC
            if global_epoch == args.num_epochs - 1:
                y_true_final = y_test_tensor.cpu().numpy()
                y_score_final = y_prob1

    return test_accs, test_precisions, test_recalls, test_f1s, y_true_final, y_score_final


# 超参数
input_dim = 30
output_dim = 2

# 差分隐私参数
target_epsilon = 1.0
target_delta = 1e-5
max_grad_norm = 2.0

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    figures_save_dir = f"figures/{args.method}"
    os.makedirs(figures_save_dir, exist_ok=True)  # 如果目录不存在就新建

    seeds = [21, 41, 13, 0, 8]


    # ---- 多次实验封装：返回四个指标矩阵 + ROC原始分数列表 ----
    def run_experiments(use_ours, use_dp=True):
        all_accs, all_precisions, all_recalls, all_f1s = [], [], [], []
        ys_true, ys_score = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            accs, precisions, recalls, f1s, y_true, y_score = main(use_ours, args, use_dp=use_dp)
            all_accs.append(accs)
            all_precisions.append(precisions)
            all_recalls.append(recalls)
            all_f1s.append(f1s)
            ys_true.append(y_true)
            ys_score.append(y_score)  # ys_* 是每个seed的数组列表（长度 = len(seeds)）
        return np.array(all_accs), np.array(all_precisions), np.array(all_recalls), np.array(
            all_f1s), ys_true, ys_score  # shape = (len(seeds), global_epochs)


    # 三种方法
    ours_accs, ours_precisions, ours_recalls, ours_f1s, ours_y_true_list, ours_y_score_list = run_experiments(True,
                                                                                                              use_dp=True)
    other_accs, other_precisions, other_recalls, other_f1s, other_y_true_list, other_y_score_list = run_experiments(
        False, use_dp=True)
    original_accs, original_precisions, original_recalls, original_f1s, orig_y_true_list, orig_y_score_list = run_experiments(
        False, use_dp=False)

    steps = np.arange(1, args.num_epochs + 1)


    # ===== 折线图（单次实验） =====
    def plot_figure1(ours, other, original, ylabel, seed):
        plt.figure(figsize=(8, 6))
        plt.plot(steps, ours, label=args.method + '+Ours', linewidth=2, color=Ours_color, linestyle='--',
                 marker='o')
        plt.plot(steps, other, label=args.method, linewidth=2, color=Others_color, linestyle='--', marker='^')
        plt.plot(steps, original, label='ε → ∞(Standard SGD)', linewidth=2, color=Original_color, linestyle='--',
                 marker='*')

        # plt.title(f"method={args.method}, seed={seed}, cosine_threshold={args.cosine_threshold}, sigma={args.sigma}",
        #           fontsize=14)
        plt.xlabel("Communication Round", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        # plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{figures_save_dir}/{args.method}_seed{seed}_{ylabel}.pdf", format="pdf", bbox_inches="tight")
        plt.show()


    for i in range(len(seeds)):
        plot_figure1(ours_accs[i], other_accs[i], original_accs[i], 'Test Accuracy', seeds[i])
        # plot_figure(ours_precisions[i], other_precisions[i], original_precisions[i], 'Test Precision', seeds[i])
        # plot_figure(ours_recalls[i], other_recalls[i], original_recalls[i], 'Test Recall', seeds[i])
        plot_figure1(ours_f1s[i], other_f1s[i], original_f1s[i], 'Test F1-Score', seeds[i])


    # ===== 折线图（多次实验平均 ± SEM） =====
    def plot_curve(x, mean, std, label, color, line_width=2, line_style='--'):
        plt.plot(x, mean, label=label, color=color, linewidth=line_width, linestyle=line_style)
        # plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        # ci95 = 1.96 * std / np.sqrt(len(seeds))
        # plt.fill_between(x, mean - ci95, mean + ci95, color=color, alpha=0.2)
        sem = std / np.sqrt(len(seeds))
        plt.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)


    def plot_figure2(ours, other, original, ylabel):
        ours_mean, ours_std = ours.mean(axis=0), ours.std(axis=0)
        other_mean, other_std = other.mean(axis=0), other.std(axis=0)
        original_mean, original_std = original.mean(axis=0), original.std(axis=0)
        print("Ours " + ylabel + ":")
        print(ours_mean)

        plt.figure(figsize=(8, 6))
        plot_curve(steps, ours_mean, ours_std, args.method + '+Ours', Ours_color, line_style='-')
        plot_curve(steps, other_mean, other_std, args.method, Others_color, line_style='-')
        plot_curve(steps, original_mean, original_std, 'ε → ∞(Standard SGD)', Original_color, line_style='-')

        # plt.title(f"method={args.method}, cosine_threshold={args.cosine_threshold}, sigma={args.sigma}", fontsize=14)
        plt.xlabel("Communication Round", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{figures_save_dir}/{args.method}_{ylabel}.pdf", format="pdf", bbox_inches="tight")
        plt.show()


    plot_figure2(ours_accs, other_accs, original_accs, "Average Test Accuracy")
    # plot_figure2(ours_precisions, other_precisions, original_precisions, "Average Test Precision")
    # plot_figure2(ours_recalls, other_recalls, original_recalls, "Average Test Recall")
    plot_figure2(ours_f1s, other_f1s, original_f1s, "Average Test F1-Score")


    # ======= ROC 曲线：单次实验（每个 seed 各画一张，三种方法同图对比） =======
    def plot_roc_single_run(idx):
        """
        idx: 第 idx 个 seed 的 ROC 对比图（Ours vs Other vs Original）
        """
        plt.figure(figsize=(8, 6))

        # Ours
        fpr_o, tpr_o, auc_o = roc_from_scores(ours_y_true_list[idx], ours_y_score_list[idx])
        plt.plot(fpr_o, tpr_o, label=f"{args.method}+Ours (AUC={auc_o:.3f})", linewidth=2, color=Ours_color)

        # Other
        fpr_b, tpr_b, auc_b = roc_from_scores(other_y_true_list[idx], other_y_score_list[idx])
        plt.plot(fpr_b, tpr_b, label=f"{args.method} (AUC={auc_b:.3f})", linewidth=2, color=Others_color)

        # Original
        fpr_g, tpr_g, auc_g = roc_from_scores(orig_y_true_list[idx], orig_y_score_list[idx])
        plt.plot(fpr_g, tpr_g, label=f"ε → ∞(Standard SGD) (AUC={auc_g:.3f})", linewidth=2, color=Original_color)

        # 随机参考线
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

        # plt.title(f"ROC (method={args.method}, seed={seeds[idx]})", fontsize=14)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f"{figures_save_dir}/{args.method}_seed{seeds[idx]}_ROC.pdf", format="pdf", bbox_inches="tight")
        plt.show()


    for i in range(len(seeds)):
        plot_roc_single_run(i)


    # ======= ROC 曲线：多次实验平均（mean ± SEM），三种方法各一条 =======
    def plot_roc_mean_all():
        plt.figure(figsize=(8, 6))

        # 收集每个方法的(fpr,tpr)列表
        ours_fpr_tpr = [roc_from_scores(yt, ys)[:2] for yt, ys in zip(ours_y_true_list, ours_y_score_list)]
        other_fpr_tpr = [roc_from_scores(yt, ys)[:2] for yt, ys in zip(other_y_true_list, other_y_score_list)]
        orig_fpr_tpr = [roc_from_scores(yt, ys)[:2] for yt, ys in zip(orig_y_true_list, orig_y_score_list)]

        # 聚合
        grid, ours_mean_tpr, ours_sem_tpr, ours_mean_auc = aggregate_mean_roc(ours_fpr_tpr)
        _, other_mean_tpr, other_sem_tpr, other_mean_auc = aggregate_mean_roc(other_fpr_tpr, grid)
        _, orig_mean_tpr, orig_sem_tpr, orig_mean_auc = aggregate_mean_roc(orig_fpr_tpr, grid)

        # 绘制 mean ± SEM
        def plot_mean_sem(label, color, mean_tpr, sem_tpr, mean_auc):
            plt.plot(grid, mean_tpr, label=f"{label} (AUC={mean_auc:.3f})",
                     color=color, linewidth=2)
            plt.fill_between(grid, np.maximum(mean_tpr - sem_tpr, 0.0),
                             np.minimum(mean_tpr + sem_tpr, 1.0),
                             color=color, alpha=0.2)

        plot_mean_sem(args.method + '+Ours', Ours_color, ours_mean_tpr, ours_sem_tpr, ours_mean_auc)
        plot_mean_sem(args.method, Others_color, other_mean_tpr, other_sem_tpr, other_mean_auc)
        plot_mean_sem('ε → ∞(Standard SGD)', Original_color, orig_mean_tpr, orig_sem_tpr, orig_mean_auc)

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        # plt.title('Mean ROC', fontsize=14)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f"{figures_save_dir}/{args.method}_ROC_mean.pdf", format="pdf", bbox_inches="tight")
        plt.show()


    plot_roc_mean_all()
