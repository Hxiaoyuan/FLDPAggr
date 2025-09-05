import numpy as np
from sklearn.metrics import roc_curve, auc

Ours_color = "#e55756"
Others_color = "#4c78a8"
Original_color = "#72b7b2"


# ===== ROC工具函数 =====
def roc_from_scores(y_true, y_score):
    """基于分数计算ROC曲线与AUC"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def aggregate_mean_roc(fpr_tpr_list, grid=None):
    """
    对多次实验的 ROC 曲线做均值与标准误（SEM）聚合。
    fpr_tpr_list: [(fpr, tpr), ...]
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)
    interp_tprs = []
    for fpr, tpr in fpr_tpr_list:
        interp_tpr = np.interp(grid, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
    interp_tprs = np.array(interp_tprs)
    mean_tpr = interp_tprs.mean(axis=0)
    std_tpr = interp_tprs.std(axis=0)
    sem_tpr = std_tpr / np.sqrt(len(fpr_tpr_list))
    mean_tpr[-1] = 1.0
    mean_auc = auc(grid, mean_tpr)
    return grid, mean_tpr, sem_tpr, mean_auc
