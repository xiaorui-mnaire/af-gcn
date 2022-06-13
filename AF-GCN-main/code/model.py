"""
Created on Mar 1, 2022
Pytorch Implementation of AF-GCN in
XiaoRui et al. AF-GCN: Attribute-fusing Graph Convolution Network for Recommendation

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class AFGCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(AFGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.num_useratt1 = self.dataset.n_att1s
        self.num_useratt2 = self.dataset.n_att2s
        self.num_itematt1 = self.dataset.m_att1s
        self.num_itematt2 = self.dataset.m_att2s
        self.num_itematt3 = self.dataset.m_att3s
        self.latent_dim = self.config['latent_dim_rec']  # 模型嵌入的维度
        self.n_layers = self.config['n_layers']  # 层数
        self.keep_prob = self.config['keep_prob']  # BRP损失的batch大小
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.embedding_useratt1 = torch.nn.Embedding(
            num_embeddings=self.num_useratt1, embedding_dim=self.latent_dim)
        self.embedding_useratt2 = torch.nn.Embedding(num_embeddings=self.num_useratt2, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_useratt1.weight, std=0.1)
        nn.init.normal_(self.embedding_useratt2.weight, std=0.1)
        self.embedding_itematt1 = torch.nn.Embedding(
            num_embeddings=self.num_itematt1, embedding_dim=self.latent_dim)
        self.embedding_itematt2 = torch.nn.Embedding(num_embeddings=self.num_itematt2, embedding_dim=self.latent_dim)
        self.embedding_itematt3 = torch.nn.Embedding(num_embeddings=self.num_itematt3, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_itematt1.weight, std=0.1)
        nn.init.normal_(self.embedding_itematt2.weight, std=0.1)
        nn.init.normal_(self.embedding_itematt3.weight, std=0.1)

        if self.config['pretrain'] == 0:
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.Graph_att = self.dataset.getSparseGraph_att()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")


    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for AF-GCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        useratt_emb, itematt_emb = self.getAttEmb(users_emb, items_emb)

        all_emb = torch.cat([users_emb, itematt_emb, useratt_emb, items_emb])

        embs = [torch.cat([users_emb, items_emb])]
        # embs = [all_emb]

        all_emb = torch.sparse.mm(self.Graph_att, all_emb)
        users, itematts, useratts, items = torch.split(all_emb, [self.num_users, self.num_items, self.num_users, self.num_items])  # 将user和item的嵌入分离
        all_emb = torch.cat([users, items])
        embs.append(all_emb)
        for i in range(self.n_layers-1):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)  # torch.stack()在dim维度上连接若干个张量
        light_out = torch.mean(embs, dim=1)  # 对所有层的嵌入求均值
        users, items = torch.split(light_out, [self.num_users, self.num_items])  # 将user和item的嵌入分离

        return users, items

    def getAttEmb(self, users_emb, items_emb):

        useratt1_emb = self.embedding_useratt1.weight
        useratt2_emb = self.embedding_useratt2.weight
        itematt1_emb = self.embedding_itematt1.weight
        itematt2_emb = self.embedding_itematt2.weight
        itematt3_emb = self.embedding_itematt3.weight

        useratt_emb = torch.cat([useratt1_emb, useratt2_emb])
        itematt_emb = torch.cat([itematt1_emb, itematt2_emb, itematt3_emb])

        useratt_emb = self.getFusionAttribute(users_emb, useratt_emb, torch.FloatTensor(self.dataset.UserAttNets).cuda())
        itematt_emb = self.getFusionAttribute(items_emb, itematt_emb, torch.FloatTensor(self.dataset.ItemAttNets).cuda())
        return useratt_emb, itematt_emb

    def getFusionAttribute(self, embedding, att_embedding, adj_mat):
        att_attention = self.getAttention(embedding, att_embedding, adj_mat)
        f_att_emb = torch.mm(att_attention, att_embedding)
        return f_att_emb

    def getAttention(self, embedding, att_embedding, adj_mat):
        sim_matrix = self.cosine_distance(embedding, att_embedding)
        sim_matrix = torch.mul(sim_matrix, adj_mat)
        attention = self.normalization(sim_matrix)
        return attention

    def normalization(self, matrix):
        zero_vec = -9e15 * torch.ones_like(matrix)
        attention = torch.where(matrix != 0, matrix, zero_vec)
        softmax = torch.nn.Softmax(dim=1)
        return softmax(attention)

    def cosine_distance(self, matrix1, matrix2):
        
        # matrix1_matrix2 = torch.mm(matrix1, matrix2.t())
        # matrix1_norm = torch.sqrt((torch.mul(matrix1, matrix1)).sum(axis=1))
        # matrix1_norm = matrix1_norm.reshape(1, matrix1_norm.size()[0])
        # matrix2_norm = torch.sqrt((torch.mul(matrix2, matrix2)).sum(axis=1))
        # matrix2_norm = matrix2_norm.reshape(1, matrix2_norm.size()[0])
        # cosine_distance = torch.div(matrix1_matrix2, torch.mm(matrix1_norm.t(), matrix2_norm))
        
        matrix1 = F.normalize(matrix1, p=2, dim=1)
        matrix2 = F.normalize(matrix2, p=2, dim=1)
        return torch.mm(matrix1, matrix2.t())

    def getUsersRating(self, users):  # 求users和item的偏好评级
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

