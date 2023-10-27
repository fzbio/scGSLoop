import numpy as np
import torch
from nn_data import NegativeSampling, easy_to_device
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


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
