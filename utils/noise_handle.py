import copy

import numpy as np
import cv2
import torch


def reorder_list(lst):
    # 排序列表
    sorted_lst = sorted(lst)

    # 创建一个新的空列表来存放结果
    result = [None] * len(lst)

    # 将最大值放在中间
    mid = len(lst) // 2
    result[mid] = sorted_lst.pop(len(sorted_lst) // 2)

    # 从两边插入元素
    left = mid - 1
    right = mid + 1
    for num in sorted_lst:
        if left >= 0:
            result[left] = num
            left -= 1
        elif right < len(lst):
            result[right] = num
            right += 1

    return result


def reorder_list_with_index(lst):
    # 将列表元素和原索引一起保存
    indexed_lst = [(value, idx) for idx, value in enumerate(lst)]

    # 按照值进行排序（升序）
    indexed_lst.sort(key=lambda x: x[0])

    # 创建一个新的空列表来存放结果
    result = [None] * len(lst)

    # 将最大值放在中间
    mid = len(lst) // 2
    result[mid] = indexed_lst[-1]

    # 从两边插入最小和次小的值
    left = 0
    right = len(indexed_lst)-1
    for idx in range(0, len(indexed_lst), 2):
        if idx == len(indexed_lst) - 1:  # 如果是最后一个元素，只插入一个
            if left < mid:
                result[left] = indexed_lst[left]
            elif right < len(lst):
                result[right] = indexed_lst[right]
            break
        if left < mid:
            result[left] = indexed_lst[idx]
            left += 1
        if right < len(lst):
            result[right] = indexed_lst[idx+1]
            right -= 1

    # return result
    return indexed_lst

def create_special_dict(lst, kernel_size=5):
    # 将列表元素和原索引一起保存
    indexed_lst = [(value, idx) for idx, value in enumerate(lst)]

    # 按照值进行排序（升序）
    indexed_lst.sort(key=lambda x: x[0])

    result = {}
    for ind, (_, client_ind) in enumerate(indexed_lst):

        smaller = [x for x in indexed_lst[max(0, ind-kernel_size+1):ind]]

        # 构建长度为5的列表
        while len(smaller) < kernel_size:
            smaller.append((_, client_ind))
        _m_ind = (kernel_size - 1) // 2
        value_list = [None]*kernel_size
        # value_list[_m_ind] = client_ind #中间是自己
        value_list[_m_ind] = (_, client_ind) #中间是自己

        smaller_ind = 0
        for i in range(1, _m_ind + 1):
            value_list[_m_ind - i] = smaller[smaller_ind]
            smaller_ind += 1
            value_list[_m_ind + i] = smaller[smaller_ind]
            smaller_ind += 1

        # smaller_ind = 0
        # for i in range(1, _m_ind+1):
        #     value_list[_m_ind-i] = smaller[smaller_ind][1]
        #     smaller_ind+=1
        #     value_list[_m_ind+i] = smaller[smaller_ind][1]
        #     smaller_ind+=1
        result[client_ind] = value_list
    return result


def gaussian_kernel_weights_batch(positions, center_idx, sigma, positions1=None, positions1_w=1.):
    # print(f'sigma:{sigma}')
    distances_sq = (positions - positions.gather(1, center_idx.view(-1, 1))) ** 2
    if positions1 is not None:

        # 沿 dim=1 做最大最小归一化
        min_val = positions1.min(dim=1, keepdim=True)[0]  # shape: (20, 1)
        max_val = positions1.max(dim=1, keepdim=True)[0]  # shape: (20, 1)

        positions1_norm = (positions1 - min_val) / (max_val - min_val + 1e-8)  # 防止除零

        # center_positions1 = positions1_norm[np.arange(cn), center_idx].reshape(cn, 1)
        center_positions1 = positions1_norm.gather(1, center_idx.view(-1, 1))
        distances_sq1 = (positions1_norm - center_positions1) ** 2
        distances_sq = distances_sq + distances_sq1 * positions1_w

    sigma_expanded = sigma.unsqueeze(1).unsqueeze(1)
    # 使用 sigma 计算高斯权重
    weights = torch.exp(-distances_sq / (2 * sigma_expanded ** 2))
    # print(weights)
    # 归一化权重
    normalized_weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)

    return normalized_weights

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

def selfGaussianBlur(weights, noisy_matrix):
    if isinstance(noisy_matrix, torch.Tensor):
        return torch.dot(weights, noisy_matrix)
    return np.dot(weights, noisy_matrix)


def client_vertical_noise_reduction(noisy_matrix, kernel_size=(5, 5), sigma_threshold=1, factor=1, dataset='MNIST'):

    """
        对矩阵按行方向进行高斯滤波。
        matrix: 输入矩阵，形状为 [N, M]
        kernel_size: 高斯核的大小
        sigma: 高斯核的标准差
        """
    # filtered_matrix = np.copy(noisy_matrix)
    # sigmas = []# 对每一列进行高斯滤波（沿行方向进行）
    # for col in range(noisy_matrix.shape[1]):
    #     c_p = noisy_matrix[:, col].astype(np.float32).reshape(-1, 1)
    #     sigma = min(sigma_threshold, get_layer_ada_sigma(c_p, factor))
    #     sigmas.append(sigma)
    #     # if np.abs(np.mean(c_p)) > 0.1:
    #     # 使用高斯滤波器（沿行方向）
    #     # c_p_mean = np.mean(c_p)
    #     # c_p = c_p - c_p_mean
    #     # print(c_p.reshape(-1))
    #     # filtered_matrix[:, col] = cv2.GaussianBlur(c_p, kernel_size, sigma).reshape(-1) + c_p_mean
    #     filtered_matrix[:, col] = cv2.GaussianBlur(c_p, kernel_size, sigma).reshape(-1)
    #     # print(filtered_matrix[:, col])
    # return filtered_matrix
    if dataset != 'CIFAR100':
        filtered_matrix = np.copy(noisy_matrix)
        sigmas = []# 对每一列进行高斯滤波（沿行方向进行）
        for col in range(noisy_matrix.shape[1]):
            c_p = noisy_matrix[:, col].astype(np.float32).reshape(-1, 1)
            sigma = min(sigma_threshold, get_layer_ada_sigma(c_p, factor))
            sigmas.append(sigma)
            # if np.abs(np.mean(c_p)) > 0.1:
            # 使用高斯滤波器（沿行方向）
            # c_p_mean = np.mean(c_p)
            # c_p = c_p - c_p_mean
            # print(c_p.reshape(-1))
            # filtered_matrix[:, col] = cv2.GaussianBlur(c_p, kernel_size, sigma).reshape(-1) + c_p_mean
            filtered_matrix[:, col] = cv2.GaussianBlur(c_p, kernel_size, sigma).reshape(-1)
            # print(filtered_matrix[:, col])
        return filtered_matrix
    else:
        _kernel_size = (1, kernel_size[0])
        sigma = min(sigma_threshold, get_layer_ada_mean_sigma(copy.deepcopy(noisy_matrix), factor))
        filtered_matrix = cv2.GaussianBlur(noisy_matrix, _kernel_size, sigma)
        print(sigma)
        # return filtered_matrix
        return filtered_matrix

def client_vertical_noise_reduction_2(noisy_matrix, client_sort, kernel_size=(5, 5), sigma_threshold=1, factor=1, dataset='MNIST'):
    filtered_matrix = np.copy(noisy_matrix.cpu())
    if dataset != 'CIFAR100':
        for col in range(noisy_matrix.shape[1]):
            # c_p = noisy_matrix[:, col].astype(np.float32).reshape(-1, 1)
            c_p = noisy_matrix[:, col].reshape(-1, 1)
            # sigma = min(sigma_threshold, get_layer_ada_sigma(c_p, factor))
            sigma = min(sigma_threshold, get_layer_ada_sigma(c_p, factor).item())
            for c_ind, c_list in client_sort.items():
                # positions = np.array([[i] for i, _ in enumerate(c_list)])
                positions1 = np.array([[v] for v, _ in c_list])
                positions = np.array([[idx] for idx in range(len(c_list))])
                # positions = np.array([[i] for i in c_list])
                center_idx = (len(c_list) - 1) // 2
                # weights = gaussian_kernel_weights(positions, center_idx, sigma=sigma)
                weights = gaussian_kernel_weights(positions, center_idx, sigma=sigma, positions1=positions1)
                # weights = weights_1 *0.7 + weights*0.3
                filtered_matrix[c_ind, col] = selfGaussianBlur(weights, np.array([c_p[i].cpu() for _, i in c_list])).reshape(-1)
                # filtered_matrix[c_ind, col] = selfGaussianBlur(torch.tensor(weights).float(), torch.stack([c_p[i] for _, i in c_list]).squeeze()).reshape(-1)
            # filtered_matrix[:, col] = cv2.GaussianBlur(c_p, kernel_size, sigma).reshape(-1)
            # print(filtered_matrix[:, col])
        return filtered_matrix

    else:
        # _kernel_size = (1, kernel_size[0])
        # sigma = min(sigma_threshold, get_layer_ada_mean_sigma(copy.deepcopy(noisy_matrix), factor))
        sigma = min(sigma_threshold, get_layer_ada_mean_sigma(copy.deepcopy(noisy_matrix), factor).item())
        print(f"======================({sigma})======================")
        for c_ind, c_list in client_sort.items():
            positions1 = np.array([[v] for v, _ in c_list])
            positions = np.array([[idx] for idx in range(len(c_list))])
            center_idx = (len(c_list) - 1) // 2
            # weights = gaussian_kernel_weights(positions, center_idx, sigma=sigma)
            weights = gaussian_kernel_weights(positions, center_idx, sigma=sigma, positions1=positions1)
            # weights = weights_1 *0.7 + weights*0.3
            # print(f"positions={positions}\n weights: {weights}")
            filtered_matrix[c_ind, :] = selfGaussianBlur(weights, np.array([noisy_matrix[i,:] for _, i in c_list])).reshape(-1)
            # filtered_matrix[c_ind, :] = selfGaussianBlur(torch.tensor(weights).float(), torch.stack([noisy_matrix[i,:] for _, i in c_list])).reshape(-1)
        # filtered_matrix = cv2.GaussianBlur(noisy_matrix, _kernel_size, sigma)

        # return filtered_matrix
        return filtered_matrix



def client_vertical_noise_reduction_2_batch(noisy_matrix, client_sort, kernel_size=(5, 5), sigma_threshold=1, factor=1, dataset='MNIST'):
    # filtered_matrix = np.copy(noisy_matrix)
    if dataset != 'CIFAR100':
        # sigma = min(sigma_threshold, get_layer_ada_sigma_batch(noisy_matrix, factor))
        sigma = torch.clamp(get_layer_ada_sigma_batch(noisy_matrix, factor), max=sigma_threshold)
        print(sigma)
        positions1s = []
        positionss = []
        center_idxs = []
        select_idx = []
        for c_ind, c_list in client_sort.items():
            center_idx = (len(c_list) - 1) // 2
            positions1s.append(np.array([v for v, _ in c_list]))
            positionss.append(np.array([idx for idx in range(len(c_list))]))
            center_idxs.append(center_idx)
            select_idx.append(np.array([i for _, i in c_list]))
            # weights = gaussian_kernel_weights(positions, center_idx, sigma=sigma)
        select_idx = torch.tensor(select_idx).unsqueeze(0).repeat(len(sigma), 1, 1).to(noisy_matrix.device)
        # weights = gaussian_kernel_weights_batch(np.array(positionss), center_idxs, sigma=sigma.cpu().numpy(), positions1=np.array(positions1s))
        weights = gaussian_kernel_weights_batch(torch.tensor(positionss, dtype=torch.float32).to(noisy_matrix.device),
                                                torch.tensor(center_idxs).to(noisy_matrix.device),
                                                sigma=sigma, positions1=torch.tensor(positions1s, dtype=torch.float32).to(noisy_matrix.device))
        # print(weights)
        noisy_matrix_t = noisy_matrix.T
        noisy_matrix_expanded = noisy_matrix_t.unsqueeze(1).expand(-1, 20, -1)  # (400, 20, 20)
        noisy_matrix_expanded = torch.gather(noisy_matrix_expanded, dim=2, index=select_idx)
        filtered_matrix = (noisy_matrix_expanded * weights).sum(dim=-1).T
        return filtered_matrix

    else:
        # _kernel_size = (1, kernel_size[0])
        sigma = torch.clamp(get_layer_ada_mean_sigma_batch(noisy_matrix, factor), max=sigma_threshold).view(-1).expand(noisy_matrix.shape[1])
        # sigma = min(sigma_threshold, get_layer_ada_mean_sigma(copy.deepcopy(noisy_matrix), factor).item())
        print(f"======================({sigma[0]})======================")
        positions1s = []
        positionss = []
        center_idxs = []
        select_idx = []
        for c_ind, c_list in client_sort.items():
            center_idx = (len(c_list) - 1) // 2
            positions1s.append(np.array([v for v, _ in c_list]))
            positionss.append(np.array([idx for idx in range(len(c_list))]))
            center_idxs.append(center_idx)
            select_idx.append(np.array([i for _, i in c_list]))

        select_idx = torch.tensor(select_idx).unsqueeze(0).repeat(len(sigma), 1, 1).to(noisy_matrix.device)
        # weights = gaussian_kernel_weights_batch(np.array(positionss), center_idxs, sigma=sigma.cpu().numpy(), positions1=np.array(positions1s))
        weights = gaussian_kernel_weights_batch(torch.tensor(positionss, dtype=torch.float32).to(noisy_matrix.device),
                                                torch.tensor(center_idxs).to(noisy_matrix.device),
                                                sigma=sigma,
                                                positions1=torch.tensor(positions1s, dtype=torch.float32).to(
                                                    noisy_matrix.device))
        noisy_matrix_t = noisy_matrix.T
        noisy_matrix_expanded = noisy_matrix_t.unsqueeze(1).expand(-1, 20, -1)  # (400, 20, 20)
        noisy_matrix_expanded = torch.gather(noisy_matrix_expanded, dim=2, index=select_idx)
        filtered_matrix = (noisy_matrix_expanded * weights).sum(dim=-1).T
        return filtered_matrix

def get_noise_reduction(delta_matrix_main, client_sort, kernel_size=5, sigma_thre=1, factor=1, dataset='MNIST', sort_method=1, device='cpu'):
    result = copy.copy(delta_matrix_main)
    for n in delta_matrix_main[0].keys():
        if sort_method == 2:
            p_result = client_vertical_noise_reduction_2(torch.tensor([delta_matrix_main[idx][n] for idx in range(len(client_sort))], dtype=torch.float32).to(device), client_sort, kernel_size, sigma_thre, factor, dataset=dataset)
            # for i, idx in enumerate(client_sort):
            for i, idx in enumerate(client_sort):
                result[i][n] = p_result[i]
        else:
            p_result = client_vertical_noise_reduction(np.array([delta_matrix_main[idx][n] for idx in range(len(client_sort))]), (kernel_size, kernel_size), sigma_thre, factor, dataset=dataset)
            # for i, idx in enumerate(client_sort):
            for i, idx in enumerate(client_sort):
                result[i][n] = p_result[i]
    return result

def get_noise_reduction_batch(delta_matrix_main, client_sort, kernel_size=5, sigma_thre=1, factor=1, dataset='MNIST', sort_method=1, device='cpu'):
    result = copy.copy(delta_matrix_main)
    for n in delta_matrix_main[0].keys():
        if sort_method == 2:
            p_result = client_vertical_noise_reduction_2_batch(torch.tensor([delta_matrix_main[idx][n] for idx in range(len(client_sort))], dtype=torch.float32).to(device), client_sort, kernel_size, sigma_thre, factor, dataset=dataset)
            for i, idx in enumerate(client_sort):
                result[i][n] = p_result[i].cpu()
        else:
            p_result = client_vertical_noise_reduction(np.array([delta_matrix_main[idx][n] for idx in range(len(client_sort))]), (kernel_size, kernel_size), sigma_thre, factor, dataset=dataset)
            # for i, idx in enumerate(client_sort):
            for i, idx in enumerate(client_sort):
                result[i][n] = p_result[i]
    return result


def client_noise_sort(delta_matrix_main, weight, epsilons_input, Z, batches, local_epochs, deltas_input, batches_epoch):
    client_sort = [i for i in range(len(delta_matrix_main))]
    # weight_cal = [weight[i]*Z[i]*batches[i]*local_epochs/epsilons_input[i] for i in range(len(weight))]
    weight_cal = [Z[i]*batches[i]*local_epochs*batches_epoch[i] for i in range(len(weight))]
    weight_sort = reorder_list_with_index(weight_cal)
    return [client_sort[idx] for (value, idx) in weight_sort]

def client_noise_sort2(delta_matrix_main, weight, epsilons_input, Z, batches, local_epochs, deltas_input, batches_epoch, kernel_size=5):
    # weight_cal = [weight[i]*Z[i]*batches[i]*local_epochs/epsilons_input[i] for i in range(len(weight))]
    weight_cal = [Z[i]*batches[i]*local_epochs*batches_epoch[i] for i in range(len(weight))]
    client_sort = create_special_dict(weight_cal, kernel_size)
    return client_sort

def client_noise_sort_batch(weight, Z, batches, local_epochs, deltas_input, batches_epoch, kernel_size=5):
    # weight_cal = [weight[i]*Z[i]*batches[i]*local_epochs/epsilons_input[i] for i in range(len(weight))]
    weight_cal = [Z[i]*local_epochs[i]*(batches_epoch[i]//batches[i]) for i in range(len(weight))]
    client_sort = create_special_dict(weight_cal, kernel_size)
    return client_sort


def get_layer_ada_sigma(matrix, factor=1):
    matrix = matrix.reshape(-1)
    # 最小-最大归一化到 [0, 1]
    # matrix_b = matrix / (np.max(np.abs(matrix)))
    matrix_b = matrix / (torch.max(torch.abs(matrix)))

    # 估计噪声标准差
    # estimated_sigma = np.std(matrix_b)
    estimated_sigma = torch.std(matrix_b)
    return estimated_sigma*factor

def get_layer_ada_sigma_batch(matrix, factor=1):
    # 沿 dim=1 做最大最小归一化
    x_min = matrix.min(dim=0, keepdim=True).values  # shape: (20, 1)
    x_max = matrix.max(dim=0, keepdim=True).values  # shape: (20, 1)

    x_norm = (matrix - x_min) / (x_max - x_min + 1e-8)  # 防止除零

    estimated_sigma = x_norm.std(dim=0)  # shape: (400,)
    return estimated_sigma*factor


def get_layer_ada_mean_sigma(matrix, factor=1):
    # matrix = matrix.reshape(-1)
    # 最小-最大归一化到 [0, 1]
    max_values = np.max(matrix, axis=0)
    matrix_b = matrix / max_values

    # 估计噪声标准差
    variance = np.var(matrix_b, axis=0, keepdims=True)
    estimated_sigma = np.mean(variance)
    return estimated_sigma*factor

def get_layer_ada_mean_sigma_batch(matrix, factor=1):
    # matrix = matrix.reshape(-1)
    # 最小-最大归一化到 [0, 1]
    # 沿 dim=1 做最大最小归一化
    x_min = matrix.min(dim=0, keepdim=True).values  # shape: (20, 1)
    x_max = matrix.max(dim=0, keepdim=True).values  # shape: (20, 1)

    x_norm = (matrix - x_min) / (x_max - x_min + 1e-8)  # 防止除零

    variance = x_norm.std(dim=0)  # shape: (400,)
    estimated_sigma = torch.mean(variance)
    return estimated_sigma*factor


