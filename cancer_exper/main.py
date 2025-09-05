import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from algs.moments_accountant import compute_z
from utils.privacy_params import get_epsilons_batchsizes

from opacus import PrivacyEngine
import tqdm
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
    weights = np.exp(-distances_sq / (2 * sigma ** 2))

    # 归一化权重
    normalized_weights = weights / np.sum(weights)

    return normalized_weights


def main(use_ours, use_dp=True):
    # 设置随机种子以确保可重复性
    # torch.manual_seed(21)
    # np.random.seed(21)
    torch.manual_seed(41)
    np.random.seed(41)
    torch.manual_seed(13)
    np.random.seed(13)

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
    epsilons_input, batches = get_epsilons_batchsizes(num_clients, 'Dist1')
    client_data = create_client_data(x_train_tensor, y_train_tensor, num_clients)
    method = 'Ours'

    weights = np.array([len(client_data[i]) for i in range(len(client_data))])
    deltas_input = np.array([0.0001] * num_clients)
    if method == 'epsilon_min':
        Z = [round(compute_z(epsilon=np.min(epsilons_input), dataset_size=weights[i], batch=batches[i], \
                             local_epochs=local_epochs, global_epochs=global_epochs, delta=deltas_input[i]), 2) \
             for i in range(len(epsilons_input))]
    elif method in ['Robust_HDP', 'PFA', 'DPFedAvg', 'WeiAvg', 'Ours', 'FedAvg']:
        Z = [round(compute_z(epsilon=epsilons_input[i], dataset_size=weights[i], batch=batches[i], \
                             local_epochs=local_epochs, global_epochs=global_epochs, delta=deltas_input[i]), 2) \
             for i in range(len(epsilons_input))]

    batches_epoch = [len(client_data[i][1]) for i in range(num_clients)]
    local_epochs_clients = [5] * num_clients
    client_sort_result = client_noise_sort_batch(weights, Z, batches, local_epochs_clients, batches_epoch,
                                                 kernel_size=5)

    test_accs = []
    # 联邦学习训练过程
    for global_epoch in tqdm.tqdm(range(global_epochs)):
        # print(f"\n全局训练轮次 [{global_epoch + 1}/{global_epochs}]")
        # 收集所有客户端的模型更新
        client_updates = []
        client_gradients_updates = []

        for client_id, (client_x, client_y) in enumerate(client_data):
            # print(f"  训练客户端 {client_id + 1}/{num_clients}", end="\r")

            # 每个客户端从当前全局模型开始训练
            local_model = GlobalModel(input_dim, output_dim)
            local_model.load_state_dict(global_model.state_dict())
            if use_dp:
                # 本地训练（带差分隐私）
                local_state_dict, gradients = train_local_model_with_dp(
                    local_model,
                    (client_x, client_y),
                    local_epochs,
                    learning_rate,
                    batches[client_id],
                    max_grad_norm,
                    Z[client_id]
                )
            else:
                local_state_dict, gradients = train_local_model(
                    local_model,
                    (client_x, client_y),
                    local_epochs,
                    learning_rate,
                    batches[client_id]
                )


            client_updates.append(local_state_dict)
            client_gradients_updates.append(gradients)
        if use_ours:
            similarity_matrix, client_aggregated_params = selective_federated_average(global_model, client_updates,
                                                                                      client_gradients_updates,
                                                                                      cosine_threshold, client_sort_result, sigma)
            averaged_state_dict = federated_average(global_model, client_aggregated_params)

        else:
            # 联邦平均（FedAvg）
            averaged_state_dict = federated_average(global_model, client_updates)

        # 更新全局模型
        global_model.load_state_dict(averaged_state_dict)

        # 在测试集上评估全局模型
        global_model.eval()
        with torch.no_grad():
            test_outputs = global_model(x_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test_tensor).float().mean()
            # print(f"  全局模型测试准确率: {accuracy:.4f}")
            test_accs.append(accuracy)

    # # 最终测试准确率
    # global_model.eval()
    # with torch.no_grad():
    #     test_outputs = global_model(x_test_tensor)
    #     _, predicted = torch.max(test_outputs, 1)
    #     final_accuracy = (predicted == y_test_tensor).float().mean()
    #     print(f"\n最终全局模型测试准确率: {final_accuracy:.4f}")
    return test_accs

# 超参数
input_dim = 30
output_dim = 2
num_clients = 20
global_epochs = 40
local_epochs = 5
batch_size = 4
learning_rate = 0.01

# 差分隐私参数
target_epsilon = 1.0
target_delta = 1e-5
# max_grad_norm = 1.0
max_grad_norm = 2.0
cosine_threshold = 0.1
sigma=0.4
if __name__ == '__main__':
    ours_accs = main(True)
    o_c = [a.item() for a in ours_accs]
    accs = main(False)
    c = [a.item() for a in accs]

    start_accs = main(False, use_dp=False)
    s_c = [a.item() for a in start_accs]

    print(o_c)
    print(c)
    print(s_c)
    print(f'dis:{[float(o_c[i])-float(c[i]) for i in range(len(o_c))]}')
    # accs = [0.38596490025520325, 0.38596490025520325, 0.42105263471603394, 0.5263158082962036, 0.6140350699424744, 0.5614035129547119, 0.5526315569877625, 0.6140350699424744, 0.640350878238678, 0.6491228342056274, 0.6578947305679321, 0.7017543911933899, 0.7105262875556946, 0.7368420958518982, 0.7543859481811523, 0.780701756477356, 0.7894737124443054, 0.8070175647735596, 0.8157894611358643, 0.8157894611358643, 0.8157894611358643, 0.8245614171028137, 0.8333333134651184, 0.8421052694320679, 0.8421052694320679, 0.8508771657943726, 0.8508771657943726, 0.859649121761322, 0.859649121761322, 0.859649121761322, 0.859649121761322, 0.859649121761322, 0.859649121761322, 0.8684210777282715, 0.8684210777282715, 0.8771929740905762, 0.8684210777282715, 0.8684210777282715, 0.8771929740905762, 0.8771929740905762, 0.8947368264198303, 0.8947368264198303, 0.8947368264198303, 0.8947368264198303, 0.8859649300575256, 0.8947368264198303, 0.8947368264198303, 0.9035087823867798, 0.9298245906829834, 0.9385964870452881]
    steps = [step for step in range(1, len(ours_accs)+1)]

    # print(accs)
    # 画折线图
    plt.figure(figsize=(8, 6))
    plt.plot(steps, ours_accs, label='Ours', linewidth=2, color='red', linestyle='--', marker='o')
    plt.plot(steps, accs, label='Other', linewidth=2, color='blue', linestyle='--', marker='x')
    plt.plot(steps, start_accs, label='Original', linewidth=2, linestyle='--', marker='*') # p

    # 图形美化
    plt.title(f"cosine_threshold={cosine_threshold}, sigma={sigma}", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    # plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()