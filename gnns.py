import torch
from torch_geometric.nn import SAGEConv, BatchNorm, TransformerConv, LayerNorm, Linear
from torch.nn import BatchNorm1d


# class VariationalTransformerEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = TransformerConv(in_channels, 2 * out_channels)
#         # self.bn = BatchNorm(2 * out_channels)
#         self.conv_mu = TransformerConv(2 * out_channels, out_channels)
#         self.conv_logstd = TransformerConv(2 * out_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         # x = self.bn(x)
#         return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


# class VariationalSageBatchNormEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = SAGEConv(in_channels, 2 * out_channels, aggr='mean')
#         self.bn = BatchNorm(2 * out_channels)
#         self.conv_mu = SAGEConv(2 * out_channels, out_channels, aggr='mean')
#         self.conv_logstd = SAGEConv(2 * out_channels, out_channels, aggr='mean')
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         x = self.bn(x)
#         return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
#
#
# class BatchNormMLP(torch.nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.embedding_layers = torch.nn.Sequential(
#             # torch.nn.BatchNorm1d(input_dim),
#             torch.nn.Linear(input_dim, 64),
#             # torch.nn.BatchNorm1d(64),
#             torch.nn.Dropout(p=0.2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 32),
#             # torch.nn.BatchNorm1d(32),
#             torch.nn.Dropout(p=0.2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(32, 16),
#         )
#         self.compressing_layers = torch.nn.Sequential(
#             # torch.nn.BatchNorm1d(16),
#             torch.nn.Dropout(p=0.2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(16, 1)
#         )
#
#     def forward(self, x):
#         # assert len(x.size()) == 2
#         z = self.embedding_layers(x)
#         y_logit = self.compressing_layers(z)
#         return y_logit
#
#     def embed(self, x):
#         return self.embedding_layers(x)
#
#
# class DenseBatchNormDecoder(torch.nn.Module):
#     def __init__(self, input_dim, mlp=None):
#         super().__init__()
#         if mlp is None:
#             self.mlp = BatchNormMLP(2 * input_dim)
#
#     def forward(self, z, edge_index, sigmoid=True):
#         edge_features = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
#         value = self.mlp(edge_features).view((edge_index.size()[1],))
#         return torch.sigmoid(value) if sigmoid else value
#
#     def forward_all(self, z, sigmoid=True):
#         size = z.size()
#         z1 = z.repeat(size[0], 1)
#         z2 = z.repeat(1, size[0]).view((size[0]*size[0], size[1]))
#         edge_features = torch.cat([z1, z2], dim=-1)
#         adj = self.mlp(edge_features).view((size[0], size[0]))
#         return torch.sigmoid(adj) if sigmoid else adj
#
#     def embed(self, z, edge_index):
#         edge_features = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
#         embeddings = self.mlp.embed(edge_features)
#         return embeddings

class GraphSageEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels, aggr='mean')
        self.norm1 = LayerNorm(2 * out_channels)
        self.linear1 = Linear(2 * out_channels, 4 * out_channels)
        self.linear2 = Linear(4 * out_channels, 2 * out_channels)
        self.conv2 = SAGEConv(2 * out_channels, out_channels, aggr='mean')
        self.norm2 = LayerNorm(out_channels)
        self.linear3 = Linear(out_channels, 2 * out_channels)
        self.linear4 = Linear(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.norm1(x).relu()
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.conv2(x, edge_index)
        x = self.norm2(x).relu()
        x = self.linear3(x).relu()
        x = self.linear4(x)
        return x


class VariationalGraphSageEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels, aggr='mean')
        self.conv_mu = SAGEConv(2 * out_channels, out_channels, aggr='mean')
        self.conv_logstd = SAGEConv(2 * out_channels, out_channels, aggr='mean')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class SimpleMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding_layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
        )
        self.compressing_layers = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        z = self.embedding_layers(x)
        y_logit = self.compressing_layers(z)
        return y_logit

    def embed(self, x):
        return self.embedding_layers(x)


class DenseDecoder(torch.nn.Module):
    def __init__(self, input_dim, mlp=None):
        super().__init__()
        if mlp is None:
            self.mlp = SimpleMLP(2 * input_dim)
        else:
            self.mlp = mlp

    def forward(self, z, edge_index, sigmoid=True):
        edge_features = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        value = self.mlp(edge_features).view((edge_index.size()[1],))
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        size = z.size()
        z1 = z.repeat(size[0], 1)
        z2 = z.repeat(1, size[0]).view((size[0]*size[0], size[1]))
        edge_features = torch.cat([z1, z2], dim=-1)
        adj = self.mlp(edge_features).view((size[0], size[0]))
        return torch.sigmoid(adj) if sigmoid else adj

    def embed(self, z, edge_index):
        edge_features = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        embeddings = self.mlp.embed(edge_features)
        return embeddings

