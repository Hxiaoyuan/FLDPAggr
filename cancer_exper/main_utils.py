import torch

import numpy as np

# 联邦平均函数（处理参数名称变化）


# 联邦平均函数（处理参数名称变化）
def federated_average(global_model, client_updates):
    averaged_state_dict = {}

    # 获取全局模型的参数名称
    global_keys = list(global_model.state_dict().keys())

    for key in global_keys:
        # 收集所有客户端对应参数的值
        param_list = []
        for update in client_updates:
            if key in update:
                param_list.append(update[key])
            else:
                # 如果客户端没有这个参数（由于Opacus包装），使用全局模型的当前值
                param_list.append(global_model.state_dict()[key].clone())

        # 计算平均值
        if param_list:
            averaged_state_dict[key] = torch.stack(param_list).mean(dim=0)

    return averaged_state_dict



# 计算梯度余弦相似度
def compute_cosine_similarity(grad_dict1, grad_dict2):
    """
    计算两个梯度字典之间的余弦相似度
    """
    similarities = []
    for key in grad_dict1.keys():
        if key in grad_dict2:
            grad1 = grad_dict1[key].flatten()
            grad2 = grad_dict2[key].flatten()

            # 计算余弦相似度
            cosine_sim = torch.nn.functional.cosine_similarity(
                grad1.unsqueeze(0), grad2.unsqueeze(0), dim=1
            )
            similarities.append(cosine_sim.item())

    # 返回平均余弦相似度
    return np.mean(similarities) if similarities else 0.0


# 计算梯度（从模型参数变化推断）
def compute_gradients(initial_params, updated_params):
    """
    计算参数变化作为梯度近似
    """
    gradients = {}
    for key in initial_params.keys():
        if key in updated_params:
            gradients[key] = updated_params[key] - initial_params[key]
    return gradients


def selective_federated_average(global_model, client_params_list, client_gradients_list, threshold, client_sort_result, sigma=0.2):
    """
    基于梯度余弦相似度进行选择性聚合，使用高斯核函数计算权重
    """
    num_clients = len(client_params_list)
    similarity_matrix = np.zeros((num_clients, num_clients))

    # 计算所有客户端之间的余弦相似度
    for i in range(num_clients):
        for j in range(num_clients):
            if i != j and j not in client_sort_result[i]:
                similarity_matrix[i, j] = -1
            else:
                sim = compute_cosine_similarity(client_gradients_list[i], client_gradients_list[j])
                similarity_matrix[i, j] = sim
            # similarity_matrix[j, i] = sim

    # 为每个客户端进行选择性聚合
    client_aggregated_params = []

    for client_id in range(num_clients):
        # 找到当前客户端的相似客户端（包括自己）
        similar_clients = [client_id]  # 总是包括自己

        for other_id in range(num_clients):
            # print(similarity_matrix[client_id, other_id])
            if other_id != client_id and similarity_matrix[client_id, other_id] > threshold:
                similar_clients.append(other_id)

        # 计算高斯核权重
        weights = []
        for similar_id in similar_clients:
            if similar_id == client_id:
                # 自身权重为1（最大相似度）
                weight = 1.0
            else:
                # 使用高斯核函数计算权重：exp(-(1 - sim)^2 / (2 * sigma^2))
                similarity = similarity_matrix[client_id, similar_id]
                distance = 1.0 - similarity  # 将相似度转换为距离
                weight = np.exp(-(distance ** 2) / (2 * sigma ** 2))
            weights.append(weight)

        # 归一化权重
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        print(normalized_weights)
        # 聚合相似客户端的参数
        aggregated_params = {}
        global_keys = list(global_model.state_dict().keys())

        for key in global_keys:
            weighted_sum = None
            for idx, similar_id in enumerate(similar_clients):
                if key in client_params_list[similar_id]:
                    param_value = client_params_list[similar_id][key]
                    weight = normalized_weights[idx]

                    if weighted_sum is None:
                        weighted_sum = param_value * weight
                    else:
                        weighted_sum += param_value * weight

            if weighted_sum is not None:
                aggregated_params[key] = weighted_sum

        client_aggregated_params.append(aggregated_params)

    return similarity_matrix, client_aggregated_params


def client_noise_sort_batch(weight, Z, batches, local_epochs, batches_epoch, kernel_size=5):
    # weight_cal = [weight[i]*Z[i]*batches[i]*local_epochs/epsilons_input[i] for i in range(len(weight))]
    weight_cal = [Z[i]*local_epochs[i]*(batches_epoch[i]/batches[i]) for i in range(len(weight))]
    # 将列表元素和原索引一起保存
    indexed_lst = [(value, idx) for idx, value in enumerate(weight_cal)]

    # 按照值进行排序（升序）
    indexed_lst.sort(key=lambda x: x[0])

    result = {}
    for ind, (_, client_ind) in enumerate(indexed_lst):
        result[client_ind] = [x[1] for x in indexed_lst[0:ind]]
    return result



    # 将训练数据随机分配给20个客户端
def create_client_data(x_train, y_train, num_clients):
    client_data = []
    data_size = len(x_train)
    indices = np.random.permutation(data_size)
    client_size = data_size // num_clients

    for i in range(num_clients):
        start_idx = i * client_size
        end_idx = (i + 1) * client_size if i != num_clients - 1 else data_size
        client_indices = indices[start_idx:end_idx]
        client_x = x_train[client_indices]
        client_y = y_train[client_indices]
        client_data.append((client_x, client_y))

    return client_data