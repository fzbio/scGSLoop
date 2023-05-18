import pandas as pd
import torch
import numpy as np


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def slack_precision_df(label_df, pred_df, resolution):
    if len(pred_df) != 0:
        slack_label_df = get_slack_df(label_df, resolution)
        precision = len(pred_df.merge(slack_label_df, on=['chrom1', 'x1', 'y1'])) / len(pred_df)
        return precision
    else:
        return None


def slack_recall_df(label_df, pred_df, resolution):
    if len(label_df) != 0:
        slack_pred_df = get_slack_df(pred_df, resolution)
        recall = len(slack_pred_df.merge(label_df, on=['chrom1', 'x1', 'y1'])) / len(label_df)
        return recall
    else:
        return None


def slack_f1_df(label_df, pred_df, resolution):
    if len(pred_df) != 0 and len(label_df) != 0:
        precision = slack_precision_df(label_df, pred_df, resolution)
        recall = slack_recall_df(label_df, pred_df, resolution)
        if precision + recall == 0:
            return None
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
    else:
        return None


def slack_metrics_df(label_df, pred_df, resolution):
    recall = slack_recall_df(label_df, pred_df, resolution)
    precision = slack_precision_df(label_df, pred_df, resolution)
    if recall is None or precision is None or recall + precision == 0:
        f1 = None
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def get_slack_df(df, resolution):
    df = df.drop(['x2', 'chrom2', 'y2'], axis=1)
    dfs = []
    for row_offset in range(-2, 3):
        for col_offset in range(-2, 3):
            current_df = df.copy()
            current_df['x1'] = current_df['x1'] + row_offset * resolution
            current_df['y1'] = current_df['y1'] + col_offset * resolution
            current_df = current_df[(current_df['x1'] >= 0) & (current_df['y1'] >= 0)]
            dfs.append(current_df)
    final_df = pd.concat(dfs)
    final_df = final_df.drop_duplicates()
    return final_df


@torch.no_grad()
def slack_f1_evaluate_all(model, loader, device, threshold):
    model.eval()
    f1all = []
    for batch in loader:
        x = batch['image'].to(device)
        y_pred = model(x, sigmoid=True)
        y_label = batch['label'].to(device)
        f1 = slack_f1_batch(y_pred, y_label, threshold)
        if f1 is not None:
            f1all.append(f1)
    # print(len(f1all))
    return np.mean(np.asarray(f1all, dtype='float'))


def slack_f1_batch(pred, label, threshold):
    pred = pred.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    pred = (pred > threshold).astype('int')
    label = label.astype('int')
    pred = pred.reshape(pred.shape[0], pred.shape[-2], pred.shape[-1])
    label = label.reshape(label.shape[0], label.shape[-2], label.shape[-1])
    pred = np.nonzero(pred)
    label = np.nonzero(label)
    pred = set(zip(pred[0], pred[1], pred[2]))
    label = set(zip(label[0], label[1], label[2]))
    slack_pred = get_slack_coords(pred)
    slack_label = get_slack_coords(label)
    if len(pred) != 0 and len(label) != 0:
        precision = len(pred & slack_label) / len(pred)
        recall = len(slack_pred & label) / len(label)
        if precision + recall == 0:
            return None
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
    else:
        return None


def get_slack_coords(coord_set):
    slack_set = set()
    for coord in coord_set:
        for row_offset in range(-2, 3):
            for col_offset in range(-2, 3):
                i, j = coord[1] + row_offset, coord[2] + col_offset
                if i >= 0 and j >= 0:
                    slack_set.add((coord[0], i, j))
    return slack_set


class BinaryThresholdConfusionMatrix(object):
    def __init__(self, thresholds, device=torch.device("cpu")):
        self.thresholds = thresholds.to(device)
        self.confusion_matrix = torch.zeros(4, len(thresholds), dtype=torch.long, device=device)
        self.device = device

    def reset(self):
        self.confusion_matrix.fill_(0)

    def update(self, output):
        preds, labels = output  # unpack
        preds, labels = preds.detach(), labels.detach()  # get off graph
        if preds.device != self.device:
            preds = preds.to(self.device)  # coord device
        if labels.device != self.device:
            labels = labels.to(self.device)  # coord device
        preds, labels = preds.view(-1), labels.view(-1)  # flatten
        preds, locs = torch.sort(preds)  # sort
        labels = torch.cumsum(labels[locs], 0)  # pool for reuse
        labels = torch.cat([torch.tensor([0], device=self.device), labels], dim=0)  # pre-pending 0
        changes = torch.searchsorted(preds, self.thresholds, right=True)  # get threshold change index
        neg_preds = labels[changes]  # get fwd change accumulation
        pos_preds = labels[-1] - neg_preds  # get bck change accumulation
        self.confusion_matrix[0] += (pos_preds).type(torch.long)  # TP
        self.confusion_matrix[1] += (len(labels) - 1 - changes - pos_preds).type(
            torch.long)  # FP (-1 accounts for prepend. 0)
        self.confusion_matrix[2] += (changes - neg_preds).type(torch.long)  # TN
        self.confusion_matrix[3] += (neg_preds).type(torch.long)  # FN

    def precision(self):
        cm = self.confusion_matrix
        cm = cm.type(torch.DoubleTensor)
        return cm[0] / (cm[0]+cm[1] + 1e-15)

    def recall(self):
        cm = self.confusion_matrix
        cm = cm.type(torch.DoubleTensor)
        return cm[0] / (cm[0] + cm[3] + 1e-15)

    def precision_recall(self):
        return self.precision(), self.recall()

    def prauc(self):
        precision, recall = self.precision_recall()
        auc = -1 * torch.trapz(precision, recall)
        return auc

    def roc_auc(self):
        cm = self.confusion_matrix
        cm = cm.type(torch.DoubleTensor)
        tpr = self.recall()
        fpr = cm[1] / (cm[1] + cm[2] + 1e-15)
        auc = -1 * torch.trapz(tpr, fpr)
        return auc


if __name__ == '__main__':
    pass
