import torch as th
import torch.nn as nn
from torch.nn import functional as F, Parameter
from dgl.nn.pytorch import RelGraphConv
from dgl import function as fn
import dgl
import torch
from collections import defaultdict
from utils.information_fusion import get_infomation_by_path
from utils.information_fusion import get_entity_eigenvector
from utils.information_fusion import load_entity_num
import os.path
import pickle
import numpy as np
from collections import defaultdict

class BaseRGCN(nn.Module):
    def __init__(self, args, train_entity_num, ookb_entity_num, train_triplets,aux_triplets, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=True):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

        self.fc_layer = nn.Linear(args.embedding_dim * 2, args.embedding_dim, bias=False)
        nn.init.xavier_normal_(self.fc_layer.weight, gain=1.414)
        self.max_neighbor = 64
        self.train_entity_num = train_entity_num
        self.ookb_entity_num = ookb_entity_num
        self.entity2class = get_infomation_by_path(args.entity2class_path)
        class2entity_ = get_infomation_by_path(args.class2entity_path)
        self.cluster_num = len(class2entity_)
        class2entity_ = sorted(class2entity_.items(), key=lambda class2entity: class2entity[0], reverse=False)
        self.class2entity = {}
        for i in range(len(class2entity_)):
            self.class2entity[class2entity_[i][0]] = class2entity_[i][1]

        self.rel = self.get_rel_htype_ttype(train_triplets)
        graph_train, self.train_relation, self.cnt_entity = self.get_graph(train_triplets, aux_triplets)
        self.r_PAD = self.train_relation * 2
        self.e_PAD = self.entity2class.shape[0]
        self.graph_train, self.weight_graph = self.sample_neighbor(graph_train)
        self.ookb_entity_id = self.get_ookb_entity_id(
            entity_idx_path='.\data\\fb15k\\' + args.sub_data + '\entity2id.txt')
        ookb_entity_class_probability = self.get_ookb_entity_class(args)

        # # 单一类型的ookb
        # ookb_entity_class_probability = np.asarray(ookb_entity_class_probability)
        # max_indices = np.argmax(ookb_entity_class_probability, axis=1)
        # for i in range(ookb_entity_class_probability.shape[0]):
        #     ookb_entity_class_probability[i, max_indices[i]] = 1
        #     ookb_entity_class_probability[i, ookb_entity_class_probability[i] != 1] = 0

        ookb_entity_class_probability = np.asarray(ookb_entity_class_probability)
        for i in range(ookb_entity_class_probability.shape[0]):
            ookb_entity_class_probability[i, ookb_entity_class_probability[i] != 0] = 1

        self.ookb_entity_class_probability = torch.tensor(ookb_entity_class_probability).cuda()

    def get_rel_htype_ttype(self, train_triplets):
        # 创建一个字典，用于存储每个关系ID的头实体和尾实体集合
        relations = defaultdict(lambda: {'heads': set(), 'tails': set()})
        rel= defaultdict(lambda: {'headType': set(), 'tailType': set()})

        # 读取文件
        for line in train_triplets:
            # 将实体和关系的ID转换为整数（如果它们是字符串的话）
            h_id = int(line[0])
            r_id = int(line[1])
            t_id = int(line[2])
            # 将头实体和尾实体添加到对应关系ID的集合中
            relations[r_id]['heads'].add(h_id)
            relations[r_id]['tails'].add(t_id)
        # 遍历relations字典中所有的关系ID
        for rid, entities in relations.items():
            # 遍历heads元素
            head_class = [0] * self.cluster_num
            tail_class = [0] * self.cluster_num
            for head in entities['heads']:
                head_class[self.entity2class[head]] += 1
            # 遍历tails元素
            for tail in entities['tails']:
                tail_class[self.entity2class[tail]] += 1
            max_head_index = max(enumerate(head_class), key=lambda x: x[1])[0]
            max_tail_index = max(enumerate(tail_class), key=lambda x: x[1])[0]
            rel[rid]['headType'] = max_head_index
            rel[rid]['tailType'] = max_tail_index

        return rel

    def get_ookb_entity_class(self, args):
        if os.path.exists('.\data\\fb15k\\' + args.sub_data + '\\'+ str(self.cluster_num) + 'ookb_class_weight_matrix.pickle'):
            with open('.\data\\fb15k\\' + args.sub_data + '\\'+ str(self.cluster_num) + 'ookb_class_weight_matrix.pickle', 'rb') as f:
                return pickle.load(f)
        else:
            En2cl2 = []
            for entity in self.ookb_entity_id:      # 遍历所有ookb实体
                En2cl = []                          # 统计和ookb实体相似的邻居类别出现频率
                e2c = [0] * (self.cluster_num + 1)  # 统计和ookb实体相似的邻居类别出现频率
                entity_rel = [0] * (self.train_relation * 2)    # ookb实体的一跳关系数目
                for neighbor in self.graph_train[entity]:   # 遍历每个ookb实体的周围边和邻居,neighbor[0]是边,neighbor[1]是邻居
                    if neighbor[0] < self.train_relation * 2:
                        entity_rel[neighbor[0]] += 1

                for r in range(len(entity_rel)):
                    if r < self.train_relation and entity_rel[r] > 0:  # 顺关系
                        e2c[self.rel[r]['headType']] = e2c[self.rel[r]['headType']] + entity_rel[r]
                    if r >= self.train_relation and entity_rel[r] > 0:  # 逆关系
                        e2c[self.rel[r - self.train_relation]['headType']] = e2c[self.rel[r - self.train_relation]['headType']] + entity_rel[r]
                e2c[-1] = sum(e2c[:(len(e2c) - 1)])
                En2cl.append(e2c)
                En2cl2.append([sum(column) for column in zip(*En2cl)])
            prob_matrix = [[round(x / row[-1], 2) if row[-1] != 0 else 0 for x in row[:-1]] for row in En2cl2]
            with open('.\data\\fb15k\\' + args.sub_data + '\\'+ str(self.cluster_num) + 'ookb_class_weight_matrix.pickle', 'wb') as f:
                pickle.dump(prob_matrix, f)
            return prob_matrix

    # def get_ookb_entity_class(self, args):
    #     # if os.path.exists('.\data\\fb15k\\' + args.sub_data + '\\'+ str(self.cluster_num) + 'ookb_class_weight_matrix.pickle'):
    #     #     with open('.\data\\fb15k\\' + args.sub_data + '\\'+ str(self.cluster_num) + 'ookb_class_weight_matrix.pickle', 'rb') as f:
    #     #         return pickle.load(f)
    #     # else:
    #         En2cl2 = []
    #         for entity in self.ookb_entity_id:      # 遍历所有ookb实体
    #             En2cl = []                          # 统计和ookb实体相似的邻居类别出现频率
    #             for neighbor in self.graph_train[entity]:   # 遍历每个ookb实体的周围边和邻居,neighbor[0]是边,neighbor[1]是邻居
    #                 e2c = [0] * (self.cluster_num + 1)      # 统计和ookb实体相似的邻居类别出现频率
    #                 if neighbor[0] == self.train_relation * 2:
    #                     En2cl.append(e2c)
    #                     break
    #                 else:
    #                     if neighbor[0] < self.train_relation:
    #                         e = self.get_entity_neighbor(neighbor[1], neighbor[0] + self.train_relation)   # e<10036
    #                         for i in e:
    #                             e2c[int(self.entity2class[i])] = e2c[int(self.entity2class[i])] + 1
    #                         e2c[self.cluster_num] = len(e)      # 最后一列存所有类出现的总数，便于后续求类别出现频率
    #                     else:
    #                         e = self.get_entity_neighbor(neighbor[1], neighbor[0] - self.train_relation)  # e<10036
    #                         for i in e:
    #                             e2c[int(self.entity2class[i])] = e2c[int(self.entity2class[i])] + 1
    #                         e2c[self.cluster_num] = len(e)
    #                 En2cl.append(e2c)
    #             En2cl2.append([sum(column) for column in zip(*En2cl)])
    #         prob_matrix = [[round(x / row[-1], 2) if row[-1] != 0 else 0 for x in row[:-1]] for row in En2cl2]
    #         with open('.\data\\fb15k\\' + args.sub_data + '\\'+ str(self.cluster_num) + 'ookb_class_weight_matrix.pickle', 'wb') as f:
    #             pickle.dump(prob_matrix, f)
    #         return prob_matrix

    def get_entity_neighbor(self, entity, relation):
        e = []
        for neighbor in self.graph_train[entity]:
            if neighbor[0] == relation and int(neighbor[1]) < self.train_entity_num:
                e.append(neighbor[1])
        return e

    def get_ookb_entity_id(self, entity_idx_path):
        # 获得训练集中非ookb实体的长度，head-10为10336
        length = self.train_entity_num

        seen_triplets = set()
        with open(entity_idx_path, 'r') as file:
            for line in file:
                fid, tid = line.strip().split('\t')
                if int(tid) >= length:
                    seen_triplets.add(int(tid))
        lis = list(seen_triplets)
        return lis

    def get_graph(self, train_triplets, aux_triplets):
        triplet_train = []
        triplet_aux = []
        graph = defaultdict(list)
        train_entity = {}
        cnt_entity = 0
        cnt_relation = 0
        for triplet in train_triplets:
            triplet_train.append(triplet)
            train_entity[triplet[0]] = 1
            train_entity[triplet[2]] = 1
            if triplet[0] >= cnt_entity:
                cnt_entity = triplet[0] + 1
            if triplet[2] >= cnt_entity:
                cnt_entity = triplet[2] + 1
            if triplet[1] >= cnt_relation:
                cnt_relation = triplet[1] + 1

        for triplet in aux_triplets:
            if triplet[1] >= cnt_relation:
                continue
            triplet_aux.append(triplet)
            if triplet[0] >= cnt_entity:
                cnt_entity = triplet[0] + 1
            if triplet[2] >= cnt_entity:
                cnt_entity = triplet[2] + 1


        for triplet in triplet_train:
            head, relation, tail = triplet
            graph[head].append([relation, tail, 0.])
            graph[tail].append([relation + cnt_relation, head, 0.])

        self.count_imply(graph, cnt_relation)

        for triplet in triplet_aux:
            head, relation, tail = triplet
            if not head in train_entity and tail in train_entity:  # 若头实体是ookb实体
                graph[head].append([relation, tail, 0.])  # 将aux中ookb实体的邻居加进去，其中邻居id都是在train中出现过的，即id都小于10336
            if not tail in train_entity and head in train_entity:  # 若尾实体是ookb实体
                graph[tail].append(
                    [relation + cnt_relation, head, 0.])  # 将aux中ookb实体的邻居加进去，其中邻居id都是在train中出现过的，即id都小于10336

        graph = self.process_graph(graph)

        return graph, cnt_relation, cnt_entity

    def process_graph(self, graph):
        for entity in graph:
            # relation_list = list(set([neighbor[0] for neighbor in graph[entity]]))
            relation_list = defaultdict(int)
            for neighbor in graph[entity]:
                relation_list[neighbor[0]] += 1
            if len(relation_list) == 1:
                continue
            for rel_i in relation_list:
                other_relation_list = [rel for rel in relation_list if rel != rel_i]
                imply_i = self.co_relation[rel_i]
                j_imply_i = imply_i[other_relation_list].max()
                for _idx, neighbor in enumerate(graph[entity]):
                    if neighbor[0] == rel_i:
                        graph[entity][_idx][2] = j_imply_i
        print ('finish processing graph')
        return graph

    def count_imply(self, graph, cnt_relation):
        co_relation = np.zeros((cnt_relation*2+1, cnt_relation*2+1), dtype=np.dtype('float32'))
        freq_relation = defaultdict(int)

        for entity in graph:
            relation_list = list(set([neighbor[0] for neighbor in graph[entity]]))
            for n_i in range(len(relation_list)):
                r_i = relation_list[n_i]
                freq_relation[r_i] += 1
                for n_j in range(n_i+1, len(relation_list)):
                    r_j = relation_list[n_j]
                    co_relation[r_i][r_j] += 1
                    co_relation[r_j][r_i] += 1

        for r_i in range(cnt_relation*2):
            co_relation[r_i] = (co_relation[r_i] * 1.0) / freq_relation[r_i]
            # co_relation[r_i][r_i] = 1.0
        self.co_relation = co_relation.transpose()
        for r_i in range(cnt_relation*2):
            co_relation[r_i][r_i] = co_relation[r_i].mean()
        print ('finish calculating co relation')

    def sample_neighbor(self, graph):
        sample_graph = np.ones((self.cnt_entity, self.max_neighbor, 2), dtype=np.dtype('int64'))
        weight_graph = np.ones((self.cnt_entity, self.max_neighbor), dtype=np.dtype('float32'))
        sample_graph[:, :, 0] *= self.r_PAD
        sample_graph[:, :, 1] *= self.e_PAD

        cnt = 0
        for entity in graph:
            num_neighbor = len(graph[entity])
            cnt += num_neighbor
            num_sample = min(num_neighbor, self.max_neighbor)
            # sample_id = random.sample(range(len(graph[entity])), num_sample)
            sample_id = range(len(graph[entity]))[:num_sample]
            # sample_graph[entity][:num_sample] = np.asarray(graph[entity])[sample_id]
            sample_graph[entity][:num_sample] = np.asarray(graph[entity])[sample_id][:, 0:2]
            weight_graph[entity][:num_sample] = np.asarray(graph[entity])[sample_id][:, 2]

        return sample_graph, weight_graph

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self) -> object:
        return None

    def forward(self, g, h, r, norm):
        hid = h
        for i, layer in enumerate(self.layers):
            if i == 0:
                entity_embedding, h = layer(g, h, r, norm)
                # entity_embedding = torch.zeros(self.num_nodes, self.h_dim).cuda()
                # entity_embedding.scatter_(0, hid.view(-1, 1).expand(-1, 200), h)
                testt = entity_embedding[:, :]
                feature_vectors = []
                for i in range(self.cluster_num):
                    feature_vectors.append(get_entity_eigenvector(testt[self.class2entity[i]]))
                feature_vectors = torch.stack(feature_vectors)
                out_entity_2 = torch.cat(
                    [feature_vectors[self.entity2class[i]].unsqueeze(0) for i in range(self.train_entity_num)], dim=0).cuda()
                all_ookb_class = torch.mm(self.ookb_entity_class_probability.to(torch.float32), feature_vectors)
                out_entity_2 = torch.cat((out_entity_2, all_ookb_class), dim=0)
                # out_entity_2 = torch.sigmoid(out_entity_2)
                class_features = torch.index_select(out_entity_2, 0, hid.squeeze())
                h = torch.cat((h, class_features), dim=1)
                h = self.fc_layer(h)
                h = F.normalize(h, p=2, dim=1)
            else:
                h = layer(g, h, r, norm)
        return h

def initializer(emb):
    emb.uniform_(-1.0, 1.0)
    return emb

class RelGraphEmbedLayer(nn.Module):
    r"""Embedding layer for featureless heterograph.
    Parameters
    ----------
    dev_id : int
        Device to run the layer.
    num_nodes : int
        Number of nodes.
    node_tides : tensor
        Storing the node type id for each node starting from 0
    num_of_ntype : int
        Number of node types
    input_size : list of int
        A list of input feature size for each node type. If None, we then
        treat certain input feature as an one-hot encoding feature.
    embed_size : int
        Output embed size
    dgl_sparse : bool, optional
        If true, use dgl.nn.NodeEmbedding otherwise use torch.nn.Embedding
    """
    def __init__(self,
                 dev_id,
                 num_nodes,
                 node_tids,
                 num_of_ntype,
                 input_size,
                 embed_size,
                 dgl_sparse=False):
        super(RelGraphEmbedLayer, self).__init__()
        self.dev_id = th.device(dev_id if dev_id >= 0 else 'cpu')
        self.embed_size = embed_size
        self.num_nodes = num_nodes
        self.dgl_sparse = dgl_sparse

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        self.node_embeds = {} if dgl_sparse else nn.ModuleDict()
        self.num_of_ntype = num_of_ntype

        for ntype in range(num_of_ntype):
            if isinstance(input_size[ntype], int):
                if dgl_sparse:
                    self.node_embeds[str(ntype)] = dgl.nn.NodeEmbedding(input_size[ntype], embed_size, name=str(ntype),
                        init_func=initializer)
                else:
                    sparse_emb = th.nn.Embedding(input_size[ntype], embed_size, sparse=True)
                    nn.init.uniform_(sparse_emb.weight, -1.0, 1.0)
                    self.node_embeds[str(ntype)] = sparse_emb
            else:
                input_emb_size = input_size[ntype].shape[1]
                embed = nn.Parameter(th.Tensor(input_emb_size, self.embed_size))
                nn.init.xavier_uniform_(embed)
                self.embeds[str(ntype)] = embed

    @property
    def dgl_emb(self):
        """
        """
        if self.dgl_sparse:
            embs = [emb for emb in self.node_embeds.values()]
            return embs
        else:
            return []

    def forward(self, node_ids, node_tids, type_ids, features):
        """Forward computation
        Parameters
        ----------
        node_ids : tensor
            node ids to generate embedding for.
        node_ids : tensor
            node type ids
        features : list of features
            list of initial features for nodes belong to different node type.
            If None, the corresponding features is an one-hot encoding feature,
            else use the features directly as input feature and matmul a
            projection matrix.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        tsd_ids = node_ids.to(self.dev_id)
        embeds = th.empty(node_ids.shape[0], self.embed_size, device=self.dev_id)
        for ntype in range(self.num_of_ntype):
            loc = node_tids == ntype
            if isinstance(features[ntype], int):
                if self.dgl_sparse:
                    embeds[loc] = self.node_embeds[str(ntype)](type_ids[loc], self.dev_id)
                else:
                    embeds[loc] = self.node_embeds[str(ntype)](type_ids[loc]).to(self.dev_id)
            else:
                embeds[loc] = features[ntype][type_ids[loc]].to(self.dev_id) @ self.embeds[str(ntype)].to(self.dev_id)

        return embeds


class RelGraphConv2(RelGraphConv):
    def forward(self, g, feat, etypes, norm=None):
        with g.local_scope():
            g.srcdata['h'] = feat
            g.edata['type'] = etypes
            if norm is not None:
                g.edata['norm'] = norm
            if self.self_loop:
                f2 = feat[:g.number_of_dst_nodes()]
                loop_weight = self.loop_weight.clone()
                loop_message = th.mm(feat, loop_weight)
            # message passing
            g.update_all(self.message_func, fn.sum(msg='msg', out='h'))
            # apply bias and activation
            node_repr = g.dstdata['h']
            if self.layer_norm:
                node_repr = self.layer_norm_weight(node_repr)
            if self.bias:
                node_repr = node_repr + self.h_bias
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.activation:
                node_repr = self.activation(node_repr)
            node_repr = self.dropout(node_repr)
            return node_repr