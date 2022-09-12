import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.utils as gutils
from utils import remove_edges, gcn_norm, edge_index_to_sparse_tensor_adj


class H2GNN(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, layer_num=2):
        super(H2GNN, self).__init__()

        self.linear1 = torch.nn.Linear(num_features, num_hidden[0])

        self.linear2 = torch.nn.Linear(num_hidden[0] + 2 * num_hidden[0] + 4 * num_hidden[0], num_classes)

        self.dropout = dropout
        self.layer_num = layer_num
        self.data = data


        temp_loop_edge_index, _ = gutils.add_self_loops(self.data.edge_index)
        sparse_adj_tensor = edge_index_to_sparse_tensor_adj(temp_loop_edge_index)


        k_hop_adjs = []
        k_hop_edge_index = []
        k_hop_adjs.append(sparse_adj_tensor)

        for i in range(self.layer_num-1):
            temp_adj_adj = torch.sparse.mm(k_hop_adjs[i], sparse_adj_tensor)

            k_hop_adjs.append(temp_adj_adj)
            k_hop_edge_index.append(temp_adj_adj._indices())

        self.k_hop_edge_index = k_hop_edge_index



        for i in range(self.layer_num-1):
            self.k_hop_edge_index[i], _ = gutils.remove_self_loops(self.k_hop_edge_index[i])
            if i == 0:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], self.data.edge_index)
            else:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], self.k_hop_edge_index[i-1])

        self.norm_adjs = []
        self.norm_adjs.append(gcn_norm(self.data.edge_index, self.data.y.shape[0]))
        self.norm_adjs.append(gcn_norm(self.k_hop_edge_index[0], self.data.y.shape[0]))


    def forward(self):
        h = self.linear1(self.data.x)
        h = F.relu(h)
        final_h = h

        # first layer
        first_hop_h = torch.sparse.mm(self.norm_adjs[0], h)
        second_hop_h = torch.sparse.mm(self.norm_adjs[1], h)
        R1 = torch.cat([first_hop_h, second_hop_h], dim=1)

        # second layer
        first_hop_h2 = torch.sparse.mm(self.norm_adjs[0], R1)
        second_hop_h2 = torch.sparse.mm(self.norm_adjs[1], R1)
        R2 = torch.cat([first_hop_h2, second_hop_h2], dim=1)

        final_h = torch.cat([final_h, R1], dim=1)
        final_h = torch.cat([final_h, R2], dim=1)
        final_h = F.dropout(final_h)
        final_h = self.linear2(final_h)

        return F.log_softmax(final_h, 1)


class H2GNN_Variant(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, layer_num=2):
        super(H2GNN_Variant, self).__init__()

        self.linear1 = torch.nn.Linear(num_features, num_hidden[0])
        self.conv1 = GCNConv(num_hidden[0], num_hidden[1], add_self_loops=False)
        self.conv2 = GCNConv(2 * num_hidden[1], num_hidden[2], add_self_loops=False)
        self.linear2 = torch.nn.Linear(num_hidden[0] + 2 * num_hidden[1] + 2 * num_hidden[2], num_classes)

        self.dropout = dropout
        self.layer_num = layer_num
        self.data = data

        temp_loop_edge_index, _ = gutils.add_self_loops(self.data.edge_index)
        sparse_adj_tensor = edge_index_to_sparse_tensor_adj(temp_loop_edge_index)

        k_hop_adjs = []
        k_hop_edge_index = []

        k_hop_adjs.append(sparse_adj_tensor)
        for i in range(self.layer_num-1):
            temp_adj_adj = torch.sparse.mm(k_hop_adjs[i], sparse_adj_tensor)
            k_hop_adjs.append(temp_adj_adj)
            k_hop_edge_index.append(temp_adj_adj._indices())

        self.k_hop_edge_index = k_hop_edge_index

        for i in range(self.layer_num-1):
            self.k_hop_edge_index[i], _ = gutils.remove_self_loops(self.k_hop_edge_index[i])
            if i == 0:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], self.data.edge_index)
            else:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], self.k_hop_edge_index[i-1])


    def forward(self):
        h = self.linear1(self.data.x)
        h = F.relu(h)
        final_h = h

        # first layer
        first_hop_h = self.conv1(h, self.data.edge_index)
        second_hop_h = self.conv1(h, self.k_hop_edge_index[0])
        R1 = torch.cat([first_hop_h, second_hop_h], dim=1)

        # second layer
        first_hop_h2 = self.conv2(R1, self.data.edge_index)
        second_hop_h2 = self.conv2(R1, self.k_hop_edge_index[0])
        R2 = torch.cat([first_hop_h2, second_hop_h2], dim=1)


        final_h = torch.cat([final_h, R1], dim=1)
        final_h = torch.cat([final_h, R2], dim=1)
        final_h = F.dropout(final_h)
        final_h = self.linear2(final_h)
        return F.log_softmax(final_h, 1)

