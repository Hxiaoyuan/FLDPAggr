import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def accuracy(model, loader, device, show=True):
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        if show:
            t = tqdm(loader)
        else:
            t = loader
        for images, target in t:
            images = images.to(device)
            target = target.to(device)
            correct += (model(images).argmax(1) == target).sum().item()
            total += target.numel()
            acc = correct / total
            if show:
                t.set_description(f'test acc: {acc*100:.2f}%')
    return acc * 100

def precision_recall_f1(model, loader, device, show=True):
    y_test = []
    y_pred = []
    model.to(device)
    with torch.no_grad():
        if show:
            t = tqdm(loader)
        else:
            t = loader
        for images, target in t:
            images = images.to(device)
            target = target.to(device)
            y_pred += model(images).argmax(1).cpu().numpy().tolist()
            y_test += target.tolist()

    return precision_score(y_test, y_pred, average='weighted'), recall_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='weighted')


def loss(model, loader, loss_fn, device, show=True):
    loss_total = 0.
    total = 0
    model.to(device)
    
    with torch.no_grad():
        if show:
            t = tqdm(loader)
        else:
            t = loader
        for images, target in t:
            images = images.to(device)
            target = target.to(device)
            #target = torch.nn.functional.one_hot(target, num_classes=10).type(torch.cuda.FloatTensor)
            if images.shape[0] == 0: # i.e. empty batch
                break
            outputs = model(images).to(device)
            loss_total += loss_fn(outputs, target) * len(target)
            total += len(target)
        
        loss_avg = loss_total / total
    return loss_avg.item()


def accuracies(models, loaders, device):
    num_clients = len(loaders)
    accs = []
    for i in range(num_clients):
        model, loader = models[i], loaders[i]
        acc = accuracy(model, loader, device, show=False)
        accs.append(acc)
    return np.array(accs)


def precision_recall_f1s(models, loaders, device):
    num_clients = len(loaders)
    precisions = []
    recalls = []
    f1s = []
    for i in range(num_clients):
        model, loader = models[i], loaders[i]
        pre, rec, f1 = precision_recall_f1(model, loader, device, show=False)
        precisions.append(pre)
        recalls.append(rec)
        f1s.append(f1)
    return np.array(precisions), np.array(recalls), np.array(f1s)

def losses(models, loaders, loss_fn, device):
    num_clients = len(models)
    losses_ = []
    for i in range(num_clients):
        model, loader = models[i], loaders[i]
        loss_ = loss(model, loader, loss_fn, device, show=False)
        losses_.append(loss_)
    return np.array(losses_)

def epsilons(priv_engines, delta):
    num_clients = len(priv_engines)
    epsilons_ = []
    for i in range(num_clients):
        epsilon = priv_engines[i].accountant.get_epsilon(delta=delta)
        epsilons_.append(round(epsilon, 3))
    return epsilons_    
        