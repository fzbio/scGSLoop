import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import negative_sampling
from nn_data import NegativeSampling, easy_to_device
from torchvision.ops import sigmoid_focal_loss
from torch import sigmoid
from torch.nn.functional import binary_cross_entropy
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


def polynomial_decay_func_factory(base_lr, max_iter, power, end_lr):
    def func(iter):
        if iter <= max_iter:
            return ((base_lr - end_lr) * ((1 - float(iter) / max_iter) ** power) + end_lr) / base_lr
        else:
            return end_lr / base_lr
    return func


@torch.no_grad()
def cnn_approx_evaluate_all(model, loader, device, alpha=0.75, gamma=3, coef=4096):
    ap_metric = BinaryAveragePrecision()
    auroc_metric = BinaryAUROC()
    model.eval()
    losses = []
    roc_auc_list = []
    ap_list = []
    for batch in loader:
        x = batch['image'].to(device)
        y_pred = model(x, sigmoid=False)
        y_label = batch['label'].to(device)

        ap = ap_metric(sigmoid(y_pred.view(-1)), y_label.int().view(-1))
        auroc = auroc_metric(sigmoid(y_pred.view(-1)), y_label.int().view(-1))
        loss = coef * sigmoid_focal_loss(y_pred, y_label, reduction='mean', alpha=alpha, gamma=gamma)
        losses.append(loss.detach().item())
        roc_auc_list.append(auroc.detach().item())
        ap_list.append(ap.detach().item())
    mean_auroc = np.nanmean(np.asarray(roc_auc_list))
    mean_ap = np.nanmean(np.asarray(ap_list))
    # print(np.mean(np.asarray(kl_losses)))
    return mean_auroc, mean_ap, np.mean(np.asarray(losses))


def cnn_train_batch(data, model, optimizer, device, alpha=0.75, gamma=3, coef=4096):
    model.train()
    x = data['image'].to(device)
    y_label = data['label'].to(device)
    optimizer.zero_grad()
    y_pred = model(x, sigmoid=False)
    loss = coef * sigmoid_focal_loss(y_pred, y_label, reduction='mean', alpha=alpha, gamma=gamma)
    loss.backward()
    optimizer.step()
    return loss.detach().item()


def gnn_train_batch(data, model, optimizer, device, recon_label, kl_coef=None):
    model.train()
    assert recon_label in ['contact', 'loop']
    data = data.to(device)
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    labels = data.edge_index if recon_label == 'contact' else data.edge_label_index
    loss = model.recon_loss(z, labels)
    if kl_coef is None:
        kl_coef = 1 / data.num_nodes
    kl_loss = kl_coef * model.kl_loss()
    loss = loss + kl_loss
    loss.backward()
    optimizer.step()
    return loss.detach().item()


@torch.no_grad()
def gnn_evaluate_all(loader, model, device, recon_label, kl_coef=None):
    assert recon_label in ['contact', 'loop']
    attrs_to_remove = ['chrom_name', 'cell_name', 'cell_type', 'edge_weights']
    if recon_label == 'contact':
        attrs_to_remove.append('edge_label_index')
    model.eval()
    y_list = []
    pred_list = []
    losses = []
    for batch in loader:
        batch = easy_to_device(batch, device, attrs_to_remove)
        z = model.encode(batch.x, batch.edge_index)
        labels = batch.edge_index if recon_label == 'contact' else batch.edge_label_index

        pos_pred = model.decode(z, labels, sigmoid=True)
        neg_sampler = NegativeSampling(edge_type=recon_label)
        batch = neg_sampler(batch)
        neg_pred = model.decode(z, batch.neg_edge_index, sigmoid=True)
        pred_list.append(torch.cat([pos_pred, neg_pred], dim=0))

        pos_y = z.new_ones(labels.size(1))
        neg_y = z.new_zeros(batch.neg_edge_index.size(1))
        y_list.append(torch.cat([pos_y, neg_y], dim=0))

        loss = model.recon_loss(z, labels, batch.neg_edge_index)
        if kl_coef is None:
            kl_coef = 1 / batch.num_nodes
        loss = loss + kl_coef * model.kl_loss()
        losses.append(loss.detach().item())
    y, pred = torch.cat(y_list).detach().cpu().numpy(), torch.cat(pred_list).detach().cpu().numpy()
    return roc_auc_score(y, pred), average_precision_score(y, pred), np.mean(np.asarray(losses))


@torch.no_grad()
def gnn_approx_evaluate_all(loader, model, device, recon_label, kl_coef=None):
    ap_metric = BinaryAveragePrecision()
    auroc_metric = BinaryAUROC()
    assert recon_label in ['contact', 'loop']
    attrs_to_remove = ['chrom_name', 'cell_name', 'cell_type', 'edge_weights']
    if recon_label == 'contact':
        attrs_to_remove.append('edge_label_index')
    model.eval()
    losses = []
    kl_losses = []
    roc_auc_list = []
    ap_list = []
    for batch in loader:
        batch = easy_to_device(batch, device, attrs_to_remove)
        z = model.encode(batch.x, batch.edge_index)
        labels = batch.edge_index if recon_label == 'contact' else batch.edge_label_index
        pos_pred = model.decode(z, labels, sigmoid=True)
        neg_sampler = NegativeSampling(edge_type=recon_label)
        batch = neg_sampler(batch)
        neg_pred = model.decode(z, batch.neg_edge_index, sigmoid=True)
        pos_y = z.new_ones(labels.size(1))
        neg_y = z.new_zeros(batch.neg_edge_index.size(1))
        y_label = torch.cat([pos_y, neg_y])
        y_pred = torch.cat([pos_pred, neg_pred])
        ap = ap_metric(y_pred, y_label.int())
        auroc = auroc_metric(y_pred, y_label.int())
        loss = model.recon_loss(z, labels, batch.neg_edge_index)
        if kl_coef is None:
            kl_coef = 1 / batch.num_nodes
        kl_loss = kl_coef * model.kl_loss()
        loss = loss + kl_loss
        losses.append(loss.detach().item())
        kl_losses.append(kl_loss.detach().item())
        roc_auc_list.append(auroc.detach().item())
        ap_list.append(ap.detach().item())
    mean_auroc = np.nanmean(np.asarray(roc_auc_list))
    mean_ap = np.nanmean(np.asarray(ap_list))
    # print(np.mean(np.asarray(kl_losses)))
    return mean_auroc, mean_ap, np.mean(np.asarray(losses))


class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def save_model(epoch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss
