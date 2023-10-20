import glob
import os.path

import cooler
import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import VGAE
from torch_geometric.loader import DataLoader
from gnns import DenseDecoder, VariationalGraphSageEncoder
from tqdm.auto import tqdm
from train_utils import EarlyStopper, save_model, load_model
from nn_data import easy_to_device
from nn_data import ShortDistanceNegSampler, short_dist_neg_sampling
from torch.utils.tensorboard import SummaryWriter
import joblib
from post_process import remove_short_distance_loops
from sklearn.linear_model import LinearRegression
from configs import DEVICE, LOADER_WORKER
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torch.utils.data import RandomSampler


def estimate_chrom_loop_num_train(train_ds, run_name, model_dir, ratio=0.75, estimate_on=100):
    assert len(train_ds) % len(train_ds.chrom_names) == 0
    cell_num = len(train_ds) // len(train_ds.chrom_names)
    estimate_on = estimate_on if estimate_on <= cell_num else cell_num
    cell_ids = np.random.choice(cell_num, estimate_on, replace=False)
    chrom_sizes = []
    loop_nums = []
    for i, chrom_name in enumerate(train_ds.chrom_names):
        graph_ids = cell_ids * len(train_ds.chrom_names) + i
        chrom_size = 0
        loop_num = 0
        for id in graph_ids:
            graph = train_ds[id]
            chrom_size += graph.num_nodes
            loop_num += graph.edge_label_index.size(dim=1)
        loop_nums.append(loop_num / len(graph_ids))
        chrom_sizes.append(chrom_size / len(graph_ids))
    X = np.array(chrom_sizes).reshape([-1, 1])
    y = np.array(loop_nums)
    reg = LinearRegression()
    reg.fit(X, y)
    joblib.dump(reg, os.path.join(model_dir, f'{run_name}.pkl'))
    chrom_loopnum_dict = {}
    for i, chrom_name in enumerate(train_ds.chrom_names):
        chrom_loopnum_dict[chrom_name] = int(ratio * reg.predict(np.array([[train_ds[i].num_nodes]]))[0])
    return chrom_loopnum_dict


def estimate_chrom_loop_num_predict(test_ds, run_name, model_dir, ratio=0.75):
    assert len(test_ds) % len(test_ds.chrom_names) == 0
    reg = joblib.load(os.path.join(model_dir, f'{run_name}.pkl'))
    chrom_loopnum_dict = {}
    for i, chrom_name in enumerate(test_ds.chrom_names):
        chrom_loopnum_dict[chrom_name] = int(ratio * reg.predict(np.array([[test_ds[i].num_nodes]]))[0])
    return chrom_loopnum_dict


def write_batch_pred_to_bed(y_pred_batch, y_proba_batch, cell_names, chrom_names, left_starts, right_starts, res, outdir):
    assert len(y_pred_batch) == len(cell_names)
    cell_df_dict = {}
    for i, cell_name in enumerate(cell_names):
        y_pred = y_pred_batch[i]
        assert y_pred.shape[0] == 1
        y_pred = y_pred[0]
        chrom_name = chrom_names[i]
        y_proba = y_proba_batch[i][0]
        left_start = left_starts[i]
        right_start = right_starts[i]
        coords_tuple = np.nonzero(y_pred)
        if len(coords_tuple) == 2:
            proba_vector = y_proba[coords_tuple[0], coords_tuple[1]]
            x1_vector = left_start + coords_tuple[0] * res
            x2_vector = x1_vector + res
            y1_vector = right_start + coords_tuple[1] * res
            y2_vector = y1_vector + res
            chroms = [chrom_name] * len(proba_vector)
            current_df = pd.DataFrame({
                'chrom1': chroms, 'x1': x1_vector, 'x2': x2_vector, 'chrom2': chroms,
                'y1': y1_vector, 'y2': y2_vector, 'proba': proba_vector
            })
        else:
            current_df = pd.DataFrame({'chrom1': [], 'x1': [], 'x2': [], 'chrom2': [], 'y1': [], 'y2': [], 'proba': []})
        if cell_name not in cell_df_dict:
            cell_df_dict[cell_name] = current_df
        else:
            cell_df_dict[cell_name] = pd.concat([cell_df_dict[cell_name], current_df])
    for k in cell_df_dict:
        short_cell_name = k.split('/')[-1]
        cell_csv_path = os.path.join(outdir, f'{short_cell_name}.csv')
        cell_df_dict[k].to_csv(
            cell_csv_path, sep='\t', header=not os.path.exists(cell_csv_path), index=False, mode='a'
        )


def convert_to_utria_dir(bedpe_dir):
    bedpe_files_paths = glob.glob(os.path.join(bedpe_dir, '*.csv'))
    for bedpe_path in bedpe_files_paths:
        df = pd.read_csv(bedpe_path, sep='\t', header=0, index_col=False, dtype={'proba': 'float'})
        df = df[df['chrom1'] == df['chrom2']]

        mask = df['x1'] > df['y1']
        df.loc[mask, ['x1', 'y1']] = df.loc[mask, ['y1', 'x1']].values
        df.loc[mask, ['x2', 'y2']] = df.loc[mask, ['y2', 'x2']].values

        df = df[df['x1'] != df['y1']]  # Filter out the loops on diagonal
        df.to_csv(bedpe_path, sep='\t', header=True, index=False)


def up_lower_tria_vote(df):
    df = df[df['chrom1'] == df['chrom2']]
    triu_df = df[df['y1'] > df['x1']]
    tril_df = df[df['y1'] < df['x1']]
    tril_df = tril_df.rename(columns={'x1': 'y1', 'x2': 'y2', 'y1': 'x1', 'y2': 'x2'})
    tril_df = tril_df[['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2', 'proba']]
    df = pd.concat([triu_df, tril_df])
    df = df.groupby(['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2'], as_index=False, sort=False).mean()
    return df


def deduplicate_bedpe_dir(bedpe_dir):
    bedpe_files_paths = glob.glob(os.path.join(bedpe_dir, '*.csv'))
    for bedpe_path in bedpe_files_paths:
        df = pd.read_csv(bedpe_path, sep='\t', header=0, index_col=False, dtype={'proba': 'float'})
        try:
            df = df.groupby(['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2'], sort=True, as_index=False).mean()
        except:
            print(df)
        df.to_csv(bedpe_path, sep='\t', header=True, index=False)


class GnnLoopCaller(object):
    def __init__(self, run_id, chroms, model_path, num_feature, train_set=None, val_set=None, log_dir=None):
        self.run_id = run_id
        self.chroms = chroms
        self.train_set = train_set
        self.val_set = val_set
        self.model_path = model_path
        self.log_dir = log_dir

        # Hyper-parameters are Hard-coded here
        self.bs = 1
        self.epochs = 50
        self.in_channels, self.out_channels = num_feature, 64
        self.learning_rate = 1e-4
        self.kl_coef = None
        self.weight_decay = 1e-3

        if self.train_set is None:
            assert self.val_set is None
        else:
            assert self.val_set is not None
        if self.train_set is not None:
            self.train_loader, self.val_loader = \
                DataLoader(
                    self.train_set, self.bs, num_workers=LOADER_WORKER,
                    pin_memory=False, exclude_keys=['edge_weights', 'cell_name', 'chrom_name', 'cell_type'],
                    sampler=RandomSampler(
                        self.train_set, replacement=True,
                        num_samples=100
                    ),
                ), \
                DataLoader(
                    self.val_set, self.bs, num_workers=LOADER_WORKER, pin_memory=False,
                )

        self.model, self.optimizer = self.get_loop_calling_settings()

    def load_model(self):
        self.model, self.optimizer, *_ = load_model(self.model, self.optimizer, self.model_path)

    def get_loop_calling_settings(self):
        net = VGAE(VariationalGraphSageEncoder(self.in_channels, self.out_channels), decoder=DenseDecoder(self.out_channels))
        # model = VGAE(VariationalGraphSageEncoder(in_channels, out_channels))
        net = net.to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        return net, opt

    def recon_loss(self, z, data, neg_edge_index=None):
        EPS = 1e-15
        pos_loss = -torch.log(
            self.model.decoder(z, data.edge_label_index, sigmoid=True) + EPS).mean()
        if neg_edge_index is None:
            neg_edge_index = short_dist_neg_sampling(data.edge_label_index, data.edge_index, data.num_nodes)
        neg_loss = -torch.log(1 -
                              self.model.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def train_batch(self, data, model, optimizer, device, recon_label, kl_coef=None):
        model.train()
        data = data.to(device)
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)

        loss = self.recon_loss(z, data)
        if kl_coef is None:
            kl_coef = 1 / data.num_nodes
        kl_loss = kl_coef * model.kl_loss()
        loss = loss + kl_loss
        loss.backward()
        optimizer.step()
        return loss.detach().item()

    @torch.no_grad()
    def approx_evaluate_all(self, loader, model, device, recon_label, kl_coef=None):
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

            neg_sampler = ShortDistanceNegSampler()
            batch = neg_sampler(batch)
            neg_pred = model.decode(z, batch.neg_edge_index, sigmoid=True)
            pos_y = z.new_ones(labels.size(1))
            neg_y = z.new_zeros(batch.neg_edge_index.size(1))
            y_label = torch.cat([pos_y, neg_y])
            y_pred = torch.cat([pos_pred, neg_pred])
            ap = ap_metric(y_pred, y_label.int())
            auroc = auroc_metric(y_pred, y_label.int())
            loss = self.recon_loss(z, batch, batch.neg_edge_index)
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

    def randomly_select_labels(self, data, alpha=0.8):
        selected_num = int(data.edge_label_index.size(1) * alpha)
        random_indices = torch.randint(data.edge_label_index.size(1), (selected_num,))
        random_indices = random_indices.to(data.edge_label_index.device)
        data.edge_label_index = data.edge_label_index[:, random_indices]
        return data

    def train(self):
        writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir else None
        early_stopper = EarlyStopper(patience=5)
        for epoch in range(1, self.epochs + 1):
            epoch_loss_list = []
            for batch in tqdm(self.train_loader, leave=False, position=0, desc='Epoch {}'.format(epoch)):
                batch = self.randomly_select_labels(batch)
                loss = self.train_batch(batch, self.model, self.optimizer, DEVICE, 'loop', kl_coef=self.kl_coef)
                epoch_loss_list.append(loss)
            auc, ap, val_loss = self.approx_evaluate_all(self.val_loader, self.model, DEVICE, 'loop', kl_coef=self.kl_coef)
            mean_loss = np.array(epoch_loss_list).mean()
            print(
                f'\t Epoch: {epoch:03d}, train loss: {mean_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {auc:.4f}, Val AP: {ap:.4f}')
            if writer is not None:
                writer.add_scalar("Loss/train", mean_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("AUC/val", auc, epoch)
                writer.add_scalar("AP/val", ap, epoch)
            if early_stopper.early_stop(val_loss):
                break
        save_model(epoch, self.model, self.optimizer, loss, self.model_path)
        if writer is not None:
            writer.flush()
            writer.close()

    @torch.no_grad()
    def predict(self, output_dir, test_set, device, threshold, resolution=10000, progress_bar=True):
        model = self.model.to(device)
        model.eval()
        os.makedirs(output_dir, exist_ok=True)
        loader = DataLoader(test_set, self.bs, num_workers=LOADER_WORKER, pin_memory=False,)
        attrs_to_remove = ['chrom_name', 'cell_name', 'cell_type', 'edge_weights', 'edge_label_index']

        print('Predicting...')
        for batch in (tqdm(loader) if progress_bar else loader):
            batch = easy_to_device(batch, device, attrs_to_remove)
            z = model.encode(batch.x, batch.edge_index)
            preds = model.decode(z, batch.edge_index, sigmoid=True)
            preds = preds.detach().cpu().numpy()
            edges = batch.edge_index.cpu().numpy()
            df = self.convert_batch_preds_to_df(preds, edges, batch.chrom_name[0], resolution)
            df = remove_short_distance_loops(df)
            df = up_lower_tria_vote(df)
            df = df[df['proba'] >= threshold]
            df = df.reset_index(drop=True)
            short_cell_name = batch.cell_name[0].split('/')[-1]
            cell_csv_path = os.path.join(output_dir, f'{short_cell_name}.csv')
            df.to_csv(
                cell_csv_path, sep='\t', header=not os.path.exists(cell_csv_path), index=False, mode='a'
            )
        deduplicate_bedpe_dir(output_dir)
        print('Done!')

    def convert_batch_preds_to_df(self, preds, edges, chrom_name, resolution):
        proba_vector = preds
        x1_vector = edges[0, :] * resolution
        x2_vector = x1_vector + resolution
        y1_vector = edges[1, :] * resolution
        y2_vector = y1_vector + resolution
        chroms = [chrom_name] * len(proba_vector)
        df = pd.DataFrame({
            'chrom1': chroms, 'x1': x1_vector, 'x2': x2_vector, 'chrom2': chroms,
            'y1': y1_vector, 'y2': y2_vector, 'proba': proba_vector
        })
        df = df.astype({'x1': 'int', 'x2': 'int', 'y1': 'int', 'y2': 'int'})
        return df

