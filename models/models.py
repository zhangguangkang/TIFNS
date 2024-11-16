import torch
import torch.nn as nn
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_
from torch_geometric.nn.conv import MessagePassing
from collections import defaultdict
from utils.information_fusion import get_infomation_by_path
from utils.information_fusion import get_entity_eigenvector
from utils.util_dgl_or_pyg import uniform
import os.path
import pickle
import numpy as np

class Complex(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, num_bases):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel_real = nn.Parameter(torch.Tensor(num_relations, args.embedding_dim))
        self.emb_rel_img = nn.Parameter(torch.Tensor(num_relations, args.embedding_dim))
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.conv1 = RGCNConv(
            args.embedding_dim, args.embedding_dim, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            args.embedding_dim, args.embedding_dim, num_relations * 2, num_bases=num_bases)
        self.dropout_ratio = args.dropout

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        nn.init.xavier_uniform_(self.emb_rel_real,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.emb_rel_img,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, entity, edge_index, edge_type, edge_norm):
        e1_embedded_real = self.emb_e_real(entity.squeeze())
        e1_embedded_img = self.emb_e_img(entity.squeeze())
        e1_embedded_real = F.relu(self.conv1(e1_embedded_real, edge_index, edge_type))
        e1_embedded_real = F.dropout(e1_embedded_real, p=self.dropout_ratio)
        e1_embedded_real = self.conv2(e1_embedded_real, edge_index, edge_type)

        e1_embedded_img = F.relu(self.conv1(e1_embedded_img, edge_index, edge_type,edge_norm))
        e1_embedded_img = F.dropout(e1_embedded_img, p=self.dropout_ratio)
        e1_embedded_img = self.conv2(e1_embedded_img, edge_index, edge_type, edge_norm)

        return e1_embedded_real, e1_embedded_img

    def get_score(self, triplets, embedding_e_real, embedding_e_imag):
        embedding_real = embedding_e_real
        embedding_imag = embedding_e_imag
        relation_embedding_real = self.emb_rel_real
        relation_embedding_imag = self.emb_rel_imag
        s_real = embedding_real[triplets[:, 0]]
        s_img = embedding_imag[triplets[:, 0]]
        o_real = embedding_real[triplets[:, 2]]
        o_img = embedding_imag[triplets[:, 2]]
        r_real = relation_embedding_real[triplets[:, 1]]
        r_img = relation_embedding_imag[triplets[:, 1]]
        score = s_real*r_real*o_real + s_real*r_img*o_img + s_img*r_real*o_img + s_img*r_img*o_real
        return score

    def score_loss(self, embedding_e_real, embedding_e_imag, triplets, target):
        score = self.get_score(triplets, embedding_e_real, embedding_e_imag)
        score = torch.sum(score, dim=1)
        score = torch.sigmoid(score)
        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding_real, embedding_imag):
        return torch.mean(embedding_real.pow(2)) + torch.mean(embedding_imag.pow(2)) + torch.mean(self.emb_rel.pow(2))

class DistMult(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, num_bases):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = nn.Parameter(torch.Tensor(num_relations, args.embedding_dim))
        self.inp_drop = torch.nn.Dropout(args.input_drop)

        self.conv1 = RGCNConv(
            args.embedding_dim, args.embedding_dim, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            args.embedding_dim, args.embedding_dim, num_relations * 2, num_bases=num_bases)
        self.dropout_ratio = args.dropout

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        nn.init.xavier_uniform_(self.emb_rel,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, entity, edge_index, edge_type, edge_norm):
        e1_embedded = self.emb_e(entity.squeeze())
        e1_embedded = F.relu(self.conv1(e1_embedded, edge_index, edge_type, edge_norm))
        e1_embedded = F.dropout(e1_embedded, p=self.dropout_ratio)
        e1_embedded = self.conv2(e1_embedded, edge_index, edge_type, edge_norm)

        return e1_embedded

    def get_score(self, triplets, embedding):
        s = embedding[triplets[:, 0]]
        r = self.emb_rel[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = s * r * o
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.get_score(triplets, embedding)
        score = torch.sum(score, dim=1)
        score = torch.sigmoid(score)
        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding) + torch.mean(self.emb_rel.pow(2))

class TransE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, num_bases, train_entity_num, ookb_entity_num, train_triplets,
                 aux_triplets):
        super(TransE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = nn.Parameter(torch.Tensor(num_relations, args.embedding_dim))
        self.inp_drop = torch.nn.Dropout(args.input_drop)

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

        graph_train, self.train_relation, self.cnt_entity = self.get_graph(train_triplets, aux_triplets)
        self.r_PAD = self.train_relation * 2
        self.e_PAD = self.entity2class.shape[0]
        self.graph_train, self.weight_graph = self.sample_neighbor(graph_train)
        self.ookb_entity_id = self.get_ookb_entity_id(
            entity_idx_path='E:\python projects\OOKB\VNNet-modified\data\\fb15k\\' + args.sub_data + '\entity2id.txt')
        ookb_entity_class_probability = self.get_ookb_entity_class(args)
        self.ookb_entity_class_probability = torch.tensor(ookb_entity_class_probability).cuda()

        self.conv1 = RGCNConv(
            args.embedding_dim, args.embedding_dim, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            args.embedding_dim, args.embedding_dim, num_relations * 2, num_bases=num_bases)
        self.dropout_ratio = args.dropout

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        nn.init.xavier_uniform_(self.emb_rel,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, entity, edge_index, edge_type, edge_norm):
        e1_embedded = self.emb_e(entity.squeeze())
        e1_embedded = F.relu(self.conv1(e1_embedded, edge_index, edge_type, edge_norm))
        e1_embedded = F.dropout(e1_embedded, p=self.dropout_ratio)
        e1_embedded = self.conv2(e1_embedded, edge_index, edge_type, edge_norm)

        # 第一版(获得全部实体的类型特征out_entity_2，抽取batch内实体对应的实体类型特征进行和实体的融合)
        test = self.emb_e.weight.data
        feature_vectors = []
        for i in range(self.cluster_num):
            feature_vectors.append(get_entity_eigenvector(test[self.class2entity[i]]))
        feature_vectors = torch.stack(feature_vectors)
        out_entity_2 = torch.cat(
            [feature_vectors[self.entity2class[i]].unsqueeze(0) for i in range(self.train_entity_num)], dim=0).cuda()
        all_ookb_class = torch.mm(self.ookb_entity_class_probability, feature_vectors)
        out_entity_2 = torch.cat((out_entity_2, all_ookb_class), dim=0)
        out_entity_2 = F.sigmoid(out_entity_2)
        class_features = torch.index_select(out_entity_2, 0, entity.squeeze())
        e2_embedded = torch.cat((e1_embedded, class_features), dim=1)
        e2_embedded = self.fc_layer(e2_embedded)
        e2_embedded = F.normalize(e2_embedded, p=2, dim=1)

        return e2_embedded

    def get_ookb_entity_class(self, args):
        if os.path.exists('E:\python projects\OOKB\VNC-modified\data\\fb15k\\' + args.sub_data + '\\'+ str(self.cluster_num) + 'ookb_class_weight_matrix.pickle'):
            with open('E:\python projects\OOKB\VNC-modified\data\\fb15k\\' + args.sub_data + '\\'+ str(self.cluster_num) + 'ookb_class_weight_matrix.pickle', 'rb') as f:
                return pickle.load(f)
        else:
            En2cl2 = []
            for entity in self.ookb_entity_id:      # 遍历所有ookb实体
                En2cl = []                          # 统计和ookb实体相似的邻居类别出现频率
                for neighbor in self.graph_train[entity]:   # 遍历每个ookb实体的周围边和邻居,neighbor[0]是边,neighbor[1]是邻居
                    e2c = [0] * (self.cluster_num + 1)      # 统计和ookb实体相似的邻居类别出现频率
                    if neighbor[0] == self.train_relation * 2:
                        En2cl.append(e2c)
                        break
                    else:
                        if neighbor[0] < self.train_relation:
                            e = self.get_entity_neighbor(neighbor[1], neighbor[0] + self.train_relation)   # e<10036
                            for i in e:
                                e2c[int(self.entity2class[i])] = e2c[int(self.entity2class[i])] + 1
                            e2c[self.cluster_num] = len(e)      # 最后一列存所有类出现的总数，便于后续求类别出现频率
                        else:
                            e = self.get_entity_neighbor(neighbor[1], neighbor[0] - self.train_relation)  # e<10036
                            for i in e:
                                e2c[int(self.entity2class[i])] = e2c[int(self.entity2class[i])] + 1
                            e2c[self.cluster_num] = len(e)
                    En2cl.append(e2c)
                En2cl2.append([sum(column) for column in zip(*En2cl)])
            prob_matrix = [[round(x / row[-1], 2) if row[-1] != 0 else 0 for x in row[:-1]] for row in En2cl2]
            with open('E:\python projects\OOKB\VNC-modified\data\\fb15k\\' + args.sub_data + '\\'+ str(self.cluster_num) + 'ookb_class_weight_matrix.pickle', 'wb') as f:
                pickle.dump(prob_matrix, f)
            return prob_matrix

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


    def get_score(self, triplets, embedding):
        s = embedding[triplets[:, 0]]
        r = self.emb_e[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = s + r - o
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.get_score(triplets, embedding)
        score = torch.sum(score, dim=1)
        score = torch.sigmoid(score)
        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding) + torch.mean(self.emb_rel.pow(2))

class ConvE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, num_bases):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = nn.Parameter(torch.Tensor(num_relations, args.embedding_dim))
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
        self.emb_dim1 = args.embedding_shape1
        self.emb_dim2 = args.embedding_dim // self.emb_dim1

        self.conv = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(args.hidden_size, args.embedding_dim)
        self.conv1 = RGCNConv(
            args.embedding_dim, args.embedding_dim, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            args.embedding_dim, args.embedding_dim, num_relations * 2, num_bases=num_bases)
        self.dropout_ratio = args.dropout

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        nn.init.xavier_uniform_(self.emb_rel,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, entity, edge_index, edge_type, edge_norm):
        e1_embedded = self.emb_e(entity.squeeze())
        e1_embedded = F.relu(self.conv1(e1_embedded, edge_index, edge_type, edge_norm))
        e1_embedded = F.dropout(e1_embedded, p=self.dropout_ratio)
        e1_embedded = self.conv2(e1_embedded, edge_index, edge_type, edge_norm)

        return e1_embedded

    def get_score(self, triplets, embedding):

        s = embedding[triplets[:, 0]]
        r = self.emb_rel[triplets[:, 1]]
        o = embedding[triplets[:, 2]]

        s = s.view(-1, 1, self.emb_dim1, self.emb_dim2)
        r = r.view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([s, r], 2)

        x = self.bn0(stacked_inputs)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        score = x * o

        return score

    def score_loss(self, embedding, triplets, target):
        score = self.get_score(triplets, embedding)
        score = torch.sum(score, dim=1)
        score = torch.sigmoid(score)
        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding) + torch.mean(self.emb_rel.pow(2))


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)
