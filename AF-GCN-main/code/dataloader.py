"""
Created on Mar 1, 2022
Pytorch Implementation of AF-GCN in
XiaoRui et al. AF-GCN: Attribute-fusing Graph Convolution Network for Recommendation

Define models here
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------------------------------
class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    DoubanBook dataset
    """

    def __init__(self, config=world.config, path="../data/DoubanBook"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']  # 用于拆分大型的adj矩阵
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        self.n_att1 = 0
        self.n_att2 = 0
        self.m_att1 = 0
        self.m_att2 = 0
        self.m_att3 = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        userAtt1_file = path + '/user_group.txt'
        userAtt2_file = path + '/user_location.txt'
        itemAtt1_file = path + '/book_author.txt'
        itemAtt2_file = path + '/book_publisher.txt'
        itemAtt3_file = path + '/book_year.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        userAtt1UniqueUsers, userAtt1Attribute, userAtt1User = [], [], []
        userAtt2UniqueUsers, userAtt2Attribute, userAtt2User = [], [], []
        itemAtt1UniqueItems, itemAtt1Attribute, itemAtt1Item = [], [], []
        itemAtt2UniqueItems, itemAtt2Attribute, itemAtt2Item = [], [], []
        itemAtt3UniqueItems, itemAtt3Attribute, itemAtt3Item = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.userAtt1DataSize = 0
        self.userAtt2DataSize = 0
        self.itemAtt1DataSize = 0
        self.itemAtt2DataSize = 0
        self.itemAtt3DataSize = 0
        # ------------------train.txt----------------------------
        with open(train_file) as f:
            for l in f.readlines():  # 作为列表返回文件中的所有行，其中每一行都是列表对象中的一项
                if len(l) > 0:
                    l = l.strip('\n').split(' ')  # strip删除开头结尾的换行，split以空格分割字符串
                    items = [int(i) for i in l[1:]]  # 提取每行中itemid
                    uid = int(l[0])  # 提取每行的第一个的userid
                    trainUniqueUsers.append(uid)  # 是一个[[],[],...]。
                    trainUser.extend([uid] * len(items))  # 得到有len(items)个user的列表，*item的长度是为了与user对应
                    trainItem.extend(items)  # 得到item的列表
                    '''
                        append在列表末尾增加一个元素，[1,2,3].append([4,5])=[1,2,3,[4,5]]
                        extend在列表末尾增加另一个列表的逐个元素,[1,2,3].extend([4,5])=[1,2,3,4,5]                     
                    '''
                    if items:
                        self.m_item = max(self.m_item, max(items))  # item_id的最大值
                    self.n_user = max(self.n_user, uid)  # user_id的最大值
                    self.traindataSize += len(items)  # 训练集所有item的数量，相当于有多少条边
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        # ---------------------test.txt------------------------------
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1  # user和item的id从0开始，加一相当于得到训练加测试里item和user的数量
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        # -------------------------user_group.txt  一对多------------------------------------
        with open(userAtt1_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    atts = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    userAtt1UniqueUsers.append(uid)
                    userAtt1User.extend([uid] * len(atts))
                    userAtt1Attribute.extend(atts)
                    self.n_att1 = max(self.n_att1, max(atts))
                    self.userAtt1DataSize += len(atts)
        self.n_att1 += 1
        self.userAtt1UniqueUsers = np.array(userAtt1UniqueUsers)
        self.userAtt1User = np.array(userAtt1User)
        self.userAtt1Attribute = np.array(userAtt1Attribute)
        # --------------------user_location.txt  一对一--------------------------
        with open(userAtt2_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if len(l) > 1:
                        atts = [int(l[1])]
                        uid = int(l[0])
                        userAtt2UniqueUsers.append(uid)
                        userAtt2User.extend([uid]*len(atts))
                        userAtt2Attribute.extend(atts)
                        self.n_att2 = max(self.n_att2, max(atts))
                        self.userAtt2DataSize += len(atts)
        self.n_att2 += 1
        self.userAtt2UniqueUsers = np.array(userAtt2UniqueUsers)
        self.userAtt2User = np.array(userAtt2User)
        self.userAtt2Attribute = np.array(userAtt2Attribute)
        # --------------------item_acthor.txt  一对一--------------------------
        with open(itemAtt1_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    atts = [int(l[1])]
                    iid = int(l[0])
                    itemAtt1UniqueItems.append(iid)
                    itemAtt1Item.extend([iid]*len(atts))
                    itemAtt1Attribute.extend(atts)
                    self.m_att1 = max(self.m_att1, max(atts))
                    self.itemAtt1DataSize += len(atts)
        self.m_att1 += 1
        self.itemAtt1UniqueItems = np.array(itemAtt1UniqueItems)
        self.itemAtt1Item = np.array(itemAtt1Item)
        self.itemAtt1Attribute = np.array(itemAtt1Attribute)
        # --------------------item_publisher.txt  一对一--------------------------
        with open(itemAtt2_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    atts = [int(l[1])]
                    iid = int(l[0])
                    itemAtt2UniqueItems.append(iid)
                    itemAtt2Item.extend([iid]*len(atts))
                    itemAtt2Attribute.extend(atts)
                    self.m_att2 = max(self.m_att2, max(atts))
                    self.itemAtt2DataSize += len(atts)
        self.m_att2 += 1
        self.itemAtt2UniqueItems = np.array(itemAtt2UniqueItems)
        self.itemAtt2Item = np.array(itemAtt2Item)
        self.itemAtt2Attribute = np.array(itemAtt2Attribute)
        # --------------------item_publisher.txt  一对一--------------------------
        with open(itemAtt3_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    atts = [int(l[1])]
                    iid = int(l[0])
                    itemAtt3UniqueItems.append(iid)
                    itemAtt3Item.extend([iid]*len(atts))
                    itemAtt3Attribute.extend(atts)
                    self.m_att3 = max(self.m_att3, max(atts))
                    self.itemAtt3DataSize += len(atts)
        self.m_att3 += 1
        self.itemAtt3UniqueItems = np.array(itemAtt3UniqueItems)
        self.itemAtt3Item = np.array(itemAtt3Item)
        self.itemAtt3Attribute = np.array(itemAtt3Attribute)

        self.Graph = None
        self.Graph_att = None
        print(f"{self.trainDataSize} interactions for training")  # user-item（边） 的数量
        print(f"{self.testDataSize} interactions for testing")
        print(f"{self.userAtt1DataSize} interactions for user_att1")
        print(f"{self.userAtt2DataSize} interactions for user_att2")
        print(f"{self.itemAtt1DataSize} interactions for item_att1")
        print(f"{self.itemAtt2DataSize} interactions for item_att2")
        print(f"{self.itemAtt3DataSize} interactions for item_att3")
        # print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph  二分图
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),  # 生成一个用户项交互矩阵R
                                      shape=(self.n_user, self.m_item))
        # (users, att1s)
        self.UserAtt1Net = csr_matrix((np.ones(len(self.userAtt1User)), (self.userAtt1User, self.userAtt1Attribute)),
                                      shape=(self.n_user, self.n_att1))
        # (users, att2s)
        self.UserAtt2Net = csr_matrix((np.ones(len(self.userAtt2User)), (self.userAtt2User, self.userAtt2Attribute)),
                                      shape=(self.n_user, self.n_att2))
        # (items, att1s)
        self.ItemAtt1Net = csr_matrix((np.ones(len(self.itemAtt1Item)), (self.itemAtt1Item, self.itemAtt1Attribute)),
                                      shape=(self.m_item, self.m_att1))
        # (items, att2s)
        self.ItemAtt2Net = csr_matrix((np.ones(len(self.itemAtt2Item)), (self.itemAtt2Item, self.itemAtt2Attribute)),
                                      shape=(self.m_item, self.m_att2))
        # (items, att3s)
        self.ItemAtt3Net = csr_matrix((np.ones(len(self.itemAtt3Item)), (self.itemAtt3Item, self.itemAtt3Attribute)),
                                      shape=(self.m_item, self.m_att3))
        '''csr_matrix ((data, indices, indptr), [shape= (M, N)]) 稀疏矩阵压缩；np.ones函数返回给定形状和数据类型的新数组，其中元素的值设置为1。'''

        self.UserAttNet = np.concatenate((self.UserAtt1Net.toarray(), self.UserAtt2Net.toarray()), axis=1)
        self.ItemAttNet = np.concatenate((self.ItemAtt1Net.toarray(), self.ItemAtt2Net.toarray(), self.ItemAtt3Net.toarray()), axis=1)

        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()  # 得到保存user的度的一个行向量
        # sum(axis=1)将一个矩阵的每一行向量相加，squeeze()从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        self.users_D[self.users_D == 0.] = 1  # 对与没有交互item的user，度为0，将他们的度设置为1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()  # 得到保存item的度的一个行向量
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate  预计算
        self._allPos = self.getUserPosItems(list(range(self.n_user)))  # 没有与user交互的item
        self.__testDict = self.__build_test()  # 测试集的字典形式
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def n_att1s(self):
        return self.n_att1

    @property
    def n_att2s(self):
        return self.n_att2

    @property
    def m_att1s(self):
        return self.m_att1

    @property
    def m_att2s(self):
        return self.m_att2

    @property
    def m_att3s(self):
        return self.m_att3

    @property
    def UserAttNets(self):
        return self.UserAttNet

    @property
    def ItemAttNets(self):
        return self.ItemAttNet

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):  # 将矩阵转化成张量
        coo = X.tocoo().astype(np.float32)  # 返回矩阵的coo格式：仅存储非0数据的行、列、值；astype()转换numpy数组的数据类型
        row = torch.Tensor(coo.row).long()  # coo的行
        col = torch.Tensor(coo.col).long()  # coo的列
        index = torch.stack([row, col])  # 将两个一维的拼接成一个二维的
        data = torch.FloatTensor(coo.data)  # coo的值
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph_att(self):
        print("loading adjacency matrix")
        if self.Graph_att is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_att_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()  # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）。
                adj_mat = sp.dok_matrix((2 * (self.n_users + self.m_items), 2 * (self.n_users + self.m_items)),
                                        dtype=np.float32)  # 生成稀疏矩阵，得到一个全为0的矩阵
                adj_mat = adj_mat.tolil()  # 将此矩阵转换为列表格式。当copy=False时，数据/索引可在该矩阵和生成的lil_矩阵之间共享。
                Ru = np.identity(self.n_users)
                Ri = np.identity(self.m_items)
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, 2 * self.n_users + self.m_items:] = R
                adj_mat[2 * self.n_users + self.m_items:, :self.n_users] = R.T
                adj_mat[:self.n_users, self.n_users + self.m_items:2 * self.n_users + self.m_items] = Ru
                adj_mat[2 * self.n_users + self.m_items:, self.n_users:self.n_users + self.m_items] = Ri

                adj_mat = adj_mat.todok()  # 将此矩阵转换为键值字典格式   得到邻接矩阵A
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))  # 对邻接矩阵A行求和，得到度
                d_inv = np.power(rowsum, -0.5).flatten()  # flatten()返回一个折叠成一维的数组，再求-0.5次方
                d_inv[np.isinf(d_inv)] = 0.  # np.isinf 判断元素是否为无穷大，是则为True，否则为False，返回形状相同的布尔数组,将无穷大（度为0的行）的值换成0
                d_mat = sp.diags(d_inv)  # 生成对角的度矩阵D^-2

                norm_adj = d_mat.dot(adj_mat)  # dot矩阵乘积 D^-2 * A
                d_mat = d_mat.tolil()
                d_mat[self.n_users:self.n_users + self.m_items, self.n_users:self.n_users + self.m_items] = Ri
                d_mat[self.n_users + self.m_items:2 * self.n_users + self.m_items,
                self.n_users + self.m_items:2 * self.n_users + self.m_items] = Ru
                d_mat = d_mat.todok()
                norm_adj = norm_adj.dot(d_mat)  # D^-2 * A * D'
                norm_adj = norm_adj.tocsr()  # 以压缩稀疏行格式返回此矩阵的副本
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_att_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph_att = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph_att = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph_att = self.Graph_att.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph_att

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacenc matrix")
                s = time()  # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）。
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items),
                                        dtype=np.float32)  # 生成稀疏矩阵，得到一个全为0的矩阵
                adj_mat = adj_mat.tolil()  # 将此矩阵转换为列表格式。当copy=False时，数据/索引可在该矩阵和生成的lil_矩阵之间共享。
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()  # 将此矩阵转换为键值字典格式   得到邻接矩阵A
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))  # 对邻接矩阵A行求和，得到度
                d_inv = np.power(rowsum, -0.5).flatten()  # flatten()返回一个折叠成一维的数组，再求-0.5次方
                d_inv[np.isinf(d_inv)] = 0.  # np.isinf 判断元素是否为无穷大，是则为True，否则为False，返回形状相同的布尔数组-----将无穷大（度为0的行）的值换成0
                d_mat = sp.diags(d_inv)  # 生成对角的度矩阵D^-2

                norm_adj = d_mat.dot(adj_mat)  # dot矩阵乘积 D^-2 * A
                norm_adj = norm_adj.dot(d_mat)  # D^-2 * A * D^-2
                norm_adj = norm_adj.tocsr()  # 以压缩稀疏行格式返回此矩阵的副本
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):  # 把测试集转变成字典形式
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]  # 可以得到与当前item交互的user
            if test_data.get(user):  # 字典(Dictionary) get()方法：返回指定键的值，如果键不在字典中返回默认值 None 或者设置的默认值。
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])  w
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):  # 参数users：list(range(self.n_user))  返回user交互的item
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
