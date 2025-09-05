import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--data_dir', type=str, default='iid-10')
    parser.add_argument('--method', type=str, default='FedAvg',
                        help="used method: 'FedAvg'/'epsilon_min'/'PFA'/'Robust_HDP'/'WeiAvg'/'DPFedAvg'/'Ours' ")
    parser.add_argument('--privacy_dist', type=str, default='Dist1',
                        help="privacy preference sampling distribution: 'Dist1'/'Dist2'/'Dist3'/'Dist4'/'Dist5'/'Dist6'/'Dist7'/'Dist8'/'Dist9' ")
    parser.add_argument('--clustering_method', type=str, default='GMM', help="'GMM'/'KMeans'/'hierarchical'")
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0, help='for data loader')
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--num_local_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--save_epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--full_batch', type=bool, default=False)
    parser.add_argument('--max_per_sample_grad_norm', type=float, default=3.0)
    parser.add_argument('--delta', type=float, default=0.0001)
    parser.add_argument('--n_pca', type=int, default=1)
    parser.add_argument('--p_prime', type=int, default=200000)
    parser.add_argument('--user_max_class', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1.0)
    # ===================================
    parser.add_argument('--denoising', action='store_true')
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=0.4)
    parser.add_argument('--cosine_threshold', type=float, default=0.1)
    return parser
