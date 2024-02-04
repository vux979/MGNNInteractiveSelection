import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Linear
import numpy as np
import time
# from icecream import ic
from itertools import groupby
from collections import Counter


class MGNN_ISLayer(MessagePassing):
    def __init__(self, dim, heads, hidden_layer, device, topk, concat=True, bias=True):
        super(MGNN_ISLayer, self).__init__(aggr='mean')
        self.dim = dim
        self.heads = heads
        self.hidden_layer = hidden_layer
        self.device = device
        self.concat = concat
        self.bias = bias
        self.topk = topk
        self.lin1 = nn.Linear(dim, dim*heads)
        self.p_lin1 = nn.Linear(dim, hidden_layer)
        self.p_lin2 = nn.Linear(hidden_layer, 1)
        self.p_bias1 = Parameter(torch.Tensor(hidden_layer))
        self.p_bias2 = Parameter(torch.Tensor(1))
        self.W_a = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=0.5)
        self.att = Parameter(torch.Tensor(1, heads, dim))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * dim))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.p_bias1)
        zeros(self.p_bias2)
        zeros(self.bias)

    def forward(self, x, edge_index, batch, layer, size=None):
        '''
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        '''
        x = x.squeeze()
        return self.propagate(edge_index, x=x, size=size, batch=batch, layer=layer)

    def message(self, edge_index_i, x_i, x_j, size_i, batch, layer):
        edge_batch = batch[edge_index_i]
        ones = torch.ones_like(edge_batch)
        edge_per_graph = torch.cat((torch.LongTensor([0]).to(self.device), torch.cumsum(global_add_pool(ones, edge_batch), dim=0)))

        e_ij = torch.mul(x_i, x_j)

        p_ij = torch.sigmoid(self.p_lin2(self.act(self.p_lin1(e_ij))+self.p_bias1)+self.p_bias2)

        top_k = torch.LongTensor([]).to(self.device)
        if layer > 0:
            for index in range(len(edge_per_graph)-1):
                sub_edge = p_ij[edge_per_graph[index]: edge_per_graph[index+1]]
                k_prop = self.topk
                k = int(k_prop * len(sub_edge))
                '''
                if len(sub_edge) <= k:
                    k = len(sub_edge)
                '''
                order = torch.topk(sub_edge.squeeze(), k).indices
                sub_top = torch.zeros_like(sub_edge.squeeze())
                sub_top[order] = 1
                top_k = torch.cat((top_k, sub_top))
            p_ij = p_ij * top_k.unsqueeze(dim=1)



        temp_ij = self.act(self.lin1(e_ij))
        temp_ij = self.drop(temp_ij)
        temp_ij = temp_ij.view(-1, self.heads, self.dim)
        c_ij = F.leaky_relu(self.att * temp_ij)
        alpha = softmax(c_ij, edge_index_i, size_i).sum(dim=-1)
        e_ij = self.sigmoid(self.W_a(e_ij).unsqueeze(dim=1) * alpha.view(-1, self.heads, 1) * p_ij.unsqueeze(dim=1))
        return e_ij

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.dim)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out


class MGNN_IS(nn.Module):
    def __init__(self, args, n_features, device):
        super(MGNN_IS, self).__init__()

        self.n_features = n_features
        self.dim = args.dim
        self.device = device
        self.batch_size = args.batch_size
        self.num_user_features = args.num_user_features
        self.heads = args.heads
        self.hidden_layer = args.hidden_layer
        self.layer = args.layer
        self.rho = args.rho
        self.topk = args.topk
        self.temp = args.temp
        self.eps = args.eps
        self.cl = args.cl
        self.fl = args.fl

        self.feature_embedding = nn.Embedding(self.n_features + 2, self.dim)
        # self.feature_embedding.weight.data.normal_(0.0,0.01)
        self.inner_mgnn_is = MGNN_ISLayer(dim=self.dim, heads=self.heads, hidden_layer=self.hidden_layer, device=self.device, topk=self.topk)
        self.outer_mgnn_is = MGNN_ISLayer(dim=self.dim, heads=self.heads, hidden_layer=self.hidden_layer, device=self.device, topk=self.topk)
        if self.fl == 1:
            self.filterlayer = FilterLayer(args)
        else:
            self.filterlayer = IdentityModule()
        self.node_weight = nn.Embedding(self.n_features + 2, 1)
        self.node_weight.weight.data.normal_(0.0, 0.01)

        self.lin_node = nn.Linear(self.dim, self.dim*self.layer)
        self.lin_t1 = nn.Linear(self.dim*self.heads, self.dim)
        self.lin_t2 = nn.Linear(self.dim*self.heads, self.dim)

        self.layer_norm = nn.LayerNorm(self.dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.update_f = nn.GRU(input_size=self.dim*self.layer, hidden_size=self.dim*self.layer)
        self.g = nn.Linear(self.dim*self.layer, 1, bias=False)

    def forward(self, data, is_training=True):

        node_id = data.x.to(self.device)
        batch = data.batch
        node_w = torch.squeeze(self.node_weight(node_id))
        sum_weight = global_add_pool(node_w, batch)

        node_emb = self.feature_embedding(node_id)
        inner_edge_index = data.edge_index
        outer_edge_index = torch.transpose(data.edge_attr, 0, 1)
        outer_edge_index = self.outer_offset(batch, self.num_user_features, outer_edge_index)

        rho = self.rho
        para_alpha = rho
        para_beta = 1-rho
        if self.layer == 2:
            inner_node_message_1 = self.inner_mgnn_is(node_emb, inner_edge_index, batch, self.layer)
            outer_node_message_1 = self.outer_mgnn_is(node_emb, outer_edge_index, batch, self.layer)
            if (self.cl == 1) and is_training:
                inner_node_message_1_cl = self.lin_t1(inner_node_message_1)
                outer_node_message_1_cl = self.lin_t2(outer_node_message_1)
                inner_node_message_1 = self.drop(self.filterlayer(inner_node_message_1_cl))
                outer_node_message_1 = self.drop(self.filterlayer(outer_node_message_1_cl))
            else:
                inner_node_message_1 = self.drop(self.filterlayer(self.lin_t1(inner_node_message_1)))
                outer_node_message_1 = self.drop(self.filterlayer(self.lin_t2(outer_node_message_1)))

            inner_node_message_0 = self.inner_mgnn_is((self.layer_norm(para_alpha*inner_node_message_1.unsqueeze(dim=1) + para_beta*node_emb)), inner_edge_index, batch, self.layer)
            outer_node_message_0 = self.outer_mgnn_is((self.layer_norm(para_alpha*outer_node_message_1.unsqueeze(dim=1) + para_beta*node_emb)), outer_edge_index, batch, self.layer)
            if (self.cl == 1) and is_training:
                inner_node_message_0_cl = self.lin_t1(inner_node_message_0)
                outer_node_message_0_cl = self.lin_t2(outer_node_message_0)
                inner_node_message_0 = self.drop(self.filterlayer(inner_node_message_0_cl))
                outer_node_message_0 = self.drop(self.filterlayer(outer_node_message_0_cl))
            else:
                inner_node_message_0 = self.drop(self.filterlayer(self.lin_t1(inner_node_message_0)))
                outer_node_message_0 = self.drop(self.filterlayer(self.lin_t2(outer_node_message_0)))

            if (self.cl == 1) and is_training:
                inner_node_message_cl = torch.cat([inner_node_message_0_cl, inner_node_message_1_cl], dim=1)
                outer_node_message_cl = torch.cat([outer_node_message_0_cl, outer_node_message_1_cl], dim=1)
            inner_node_message = torch.cat([inner_node_message_0, inner_node_message_1], dim=1)
            outer_node_message = torch.cat([outer_node_message_0, outer_node_message_1], dim=1)

            if self.layer > 1:
                node_emb = self.drop(self.act(self.lin_node(node_emb)))
                '''
                if (self.cl == 1) and is_training:
                    node_emb_cl = self.perturbed(node_emb)
                '''
            # aggregate all message
            if len(outer_node_message.size()) < len(node_emb.size()):
                outer_node_message = outer_node_message.unsqueeze(1)
                inner_node_message = inner_node_message.unsqueeze(1)
                if (self.cl == 1) and is_training:
                    outer_node_message_cl = outer_node_message_cl.unsqueeze(1)
                    inner_node_message_cl = inner_node_message_cl.unsqueeze(1)

            updated_node_input = torch.cat((node_emb, inner_node_message, outer_node_message), 1)
            if (self.cl == 1) and is_training:
                updated_node_input_cl = torch.cat((node_emb, inner_node_message_cl, outer_node_message_cl),1)

            updated_node_input = torch.transpose(updated_node_input, 0, 1)
            if (self.cl == 1) and is_training:
                updated_node_input_cl = torch.transpose(updated_node_input_cl,0,1)

            gru_h0 = torch.normal(0, 0.01, (1, node_emb.size(0), self.dim*self.layer)).to(self.device)
            gru_output, hn = self.update_f(updated_node_input, gru_h0)
            if (self.cl == 1) and is_training:
                gru_output_cl, hn_cl = self.update_f(updated_node_input_cl, gru_h0)
            updated_node = gru_output[-1]
            if (self.cl == 1) and is_training:
                updated_node_cl = gru_output_cl[-1]

            new_batch = self.split_batch(batch, self.num_user_features)
            updated_graph = torch.squeeze(global_mean_pool(updated_node, new_batch))

            if (self.cl == 1) and is_training:
                updated_graph_cl = torch.squeeze(global_mean_pool(updated_node_cl, new_batch))
                updated_graph_cl_2 = updated_graph

            item_graphs, user_graphs = torch.split(updated_graph, int(updated_graph.size(0) / 2))
            if (self.cl == 1) and is_training:
                item_graphs_cl, user_graphs_cl = torch.split(updated_graph_cl, int(updated_graph_cl.size(0) / 2))
                item_graphs_cl_2, user_graphs_cl_2 = torch.split(updated_graph_cl_2, int(updated_graph_cl_2.size(0) / 2))

            y = torch.unsqueeze(torch.sum(user_graphs * item_graphs, 1) + sum_weight, 1)
            y = torch.sigmoid(y)


        elif self.layer == 1:
            inner_node_message_0 = self.inner_mgnn_is(node_emb, inner_edge_index, batch, self.layer)
            outer_node_message_0 = self.outer_mgnn_is(node_emb, outer_edge_index, batch, self.layer)
            inner_node_message_0 = self.drop(self.filterlayer(self.lin_t1(inner_node_message_0)))
            outer_node_message_0 = self.drop(self.filterlayer(self.lin_t2(outer_node_message_0)))
            inner_node_message = inner_node_message_0
            outer_node_message = outer_node_message_0
            if self.layer > 1:
                node_emb = self.drop(self.act(self.lin_node(node_emb)))
            # aggregate all message
            if len(outer_node_message.size()) < len(node_emb.size()):
                outer_node_message = outer_node_message.unsqueeze(1)
                inner_node_message = inner_node_message.unsqueeze(1)
            updated_node_input = torch.cat((node_emb, inner_node_message, outer_node_message), 1)
            updated_node_input = torch.transpose(updated_node_input, 0, 1)

            gru_h0 = torch.normal(0, 0.01, (1, node_emb.size(0), self.dim * self.layer)).to(self.device)
            gru_output, hn = self.update_f(updated_node_input, gru_h0)
            updated_node = gru_output[-1]

            new_batch = self.split_batch(batch, self.num_user_features)
            updated_graph = torch.squeeze(global_mean_pool(updated_node, new_batch))
            item_graphs, user_graphs = torch.split(updated_graph, int(updated_graph.size(0) / 2))
            y = torch.unsqueeze(torch.sum(user_graphs * item_graphs, 1) + sum_weight, 1)
            y = torch.sigmoid(y)

        if (self.cl == 1) and is_training:

            new_batch_list = new_batch.tolist()
            node_id_list = torch.squeeze(node_id).tolist()
            node_id_batch = sorted(zip(node_id_list, new_batch_list), key=lambda x: x[1])
            node_id_groups = groupby(node_id_batch, key=lambda x: x[1])
            node_id_graph = [[item[0] for item in group] for key, group in node_id_groups]
            item_graph = node_id_graph[:int(len(node_id_graph) / 2)]
            user_graph = node_id_graph[int(len(node_id_graph) / 2):]
            item_id = [row[0] for row in item_graph]
            user_id = [row[0] for row in user_graph]

            duplicate_dict_item = mark_duplicate_indices(item_id)
            duplicate_dict_user = mark_duplicate_indices(user_id)
            duplicate_item_id = find_all_duplicates_indices(item_id)
            duplicate_user_id = find_all_duplicates_indices(user_id)
            duplicate_id = merge_and_remove_duplicates(duplicate_item_id, duplicate_user_id)
            item_embedding_list_cl = []
            item_embedding_list_cl_2 = []
            for key, value in duplicate_dict_item.items():
                if len(value) > 1:
                    temp_embedding_cl = torch.zeros(item_graphs.size()[1]).cuda()
                    temp_embedding_cl_2 = torch.zeros(item_graphs.size()[1]).cuda()
                    for id in value:
                        temp_embedding_cl = temp_embedding_cl + item_graphs_cl[id]
                        temp_embedding_cl_2 = temp_embedding_cl_2 + item_graphs_cl_2[id]
                    item_embedding_list_cl.append((temp_embedding_cl/len(value)).tolist())
                    item_embedding_list_cl_2.append((temp_embedding_cl_2/len(value)).tolist())
            item_embedding_tensor_cl = torch.Tensor(item_embedding_list_cl).cuda()
            item_embedding_tensor_cl_2 = torch.Tensor(item_embedding_list_cl_2).cuda()
            user_embedding_list_cl = []
            user_embedding_list_cl_2 = []
            for key, value in duplicate_dict_user.items():
                if len(value) > 1:
                    temp_embedding_cl = torch.zeros(user_graphs.size()[1]).cuda()
                    temp_embedding_cl_2 = torch.zeros(user_graphs.size()[1]).cuda()
                    for id in value:
                        temp_embedding_cl = temp_embedding_cl + user_graphs_cl[id]
                        temp_embedding_cl_2 = temp_embedding_cl_2 + user_graphs_cl_2[id]
                    user_embedding_list_cl.append((temp_embedding_cl/len(value)).tolist())
                    user_embedding_list_cl_2.append((temp_embedding_cl_2/len(value)).tolist())
            user_embedding_tensor_cl = torch.Tensor(user_embedding_list_cl).cuda()
            user_embedding_tensor_cl_2 = torch.Tensor(user_embedding_list_cl_2).cuda()
            item_graphs_delete_cl = torch.tensor(np.delete(item_graphs_cl.clone().to('cpu').detach().numpy(),
                                                        duplicate_id, axis=0)).cuda()
            item_graphs_delete_cl_2 = torch.tensor(np.delete(item_graphs_cl_2.clone().to('cpu').detach().numpy(),
                                                           duplicate_id, axis=0)).cuda()
            user_graphs_delete_cl = torch.tensor(np.delete(user_graphs_cl.clone().to('cpu').detach().numpy(),
                                                        duplicate_id, axis=0)).cuda()
            user_graphs_delete_cl_2 = torch.tensor(np.delete(user_graphs_cl_2.clone().to('cpu').detach().numpy(),
                                                           duplicate_id, axis=0)).cuda()
            item_graphs_select_cl = torch.cat((item_graphs_delete_cl, item_embedding_tensor_cl), dim=0)
            item_graphs_select_cl_2 = torch.cat((item_graphs_delete_cl_2, item_embedding_tensor_cl_2), dim=0)
            user_graphs_select_cl = torch.cat((user_graphs_delete_cl, user_embedding_tensor_cl), dim=0)
            user_graphs_select_cl_2 = torch.cat((user_graphs_delete_cl_2, user_embedding_tensor_cl_2), dim=0)
        if is_training:
            if self.cl == 1:
                return (y, item_graphs_select_cl, item_graphs_select_cl_2, user_graphs_select_cl, user_graphs_select_cl_2)
            else:
                return y
        else:
            return y
    def split_batch(self, batch, user_node_num):
        """
        split batch id into user nodes and item nodes 
        """
        ones = torch.ones_like(batch)
        nodes_per_graph = global_add_pool(ones, batch)
        cum_num = torch.cat((torch.LongTensor([0]).to(self.device), torch.cumsum(nodes_per_graph, dim=0)[:-1]))
        cum_num_list = [cum_num + i for i in range(user_node_num)]
        multi_hot = torch.cat(cum_num_list)
        test = torch.sum(F.one_hot(multi_hot, ones.size(0)), dim=0) * (torch.max(batch) + 1)
        return batch + test

    def outer_offset(self, batch, user_node_num, outer_edge_index):
        ones = torch.ones_like(batch)
        nodes_per_graph = global_add_pool(ones, batch)
        inter_per_graph = (nodes_per_graph - user_node_num) * user_node_num * 2
        cum_num = torch.cat((torch.LongTensor([0]).to(self.device), torch.cumsum(nodes_per_graph, dim=0)[:-1]))
        offset_list = torch.repeat_interleave(cum_num, inter_per_graph, dim=0).repeat(2, 1)
        outer_edge_index_offset = outer_edge_index + offset_list
        return outer_edge_index_offset
    def perturbed(self, input_tensor, perturbed=True):
        if perturbed:
            random_noise = torch.rand_like(input_tensor).cuda()
            input_tensor = input_tensor + torch.sign(input_tensor) * F.normalize(random_noise, dim=-1) * self.eps
        return input_tensor


class FilterLayer(nn.Module):
    def __init__(self, config):
        super(FilterLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn((1, config.hidden_layer//2 + 1, 2), dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(0.1)


    def forward(self, input_tensor):
        seq_len, hidden = input_tensor.size()
        x = torch.rfft(input_tensor, signal_ndim=2, normalized=True)
        weight = self.complex_weight
        x = x * weight
        sequence_emb_fft = torch.irfft(x, signal_ndim=2, normalized=True, signal_sizes=(seq_len, hidden))
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = hidden_states + input_tensor
        # hidden_states = sequence_emb_fft
        return hidden_states


class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(self, x):
        return x


def mark_duplicate_indices(lst):
    index_dict = {}
    for i, item in enumerate(lst):
        if item in index_dict:
            index_dict[item].append(i)
        else:
            index_dict[item] = [i]
    return index_dict



def find_all_duplicates_indices(lst):
    counts = Counter(lst)
    duplicates_indices = [i for i, x in enumerate(lst) if counts[x] > 1]
    return duplicates_indices


def merge_and_remove_duplicates(list1, list2):
    merged_list = list1 + list2

    # 去除重复元素
    unique_list = list(set(merged_list))

    return unique_list