import networkx as nx
import numpy as np
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
import sys
import scipy
import sklearn
import json
from collections import defaultdict
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
import pickle as pkl
import scipy.sparse as sp
from base_model_SSL import GCN
from base_model_SSL import GraphConvolution
import time
import datetime
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score

#from torch_geometric.data import Data


def update_competency(t, T, c_0, p):
    term = pow(((1 - pow(c_0,p))*(t/T)) + pow(c_0,p), (1/p))
    return min([1,term])

# from base_model import GCN

def dropedge(adj,num_nodes,p=0.5,drop=True):


    if drop==False:
        adj = normalize(adj + sp.eye(adj.shape[0]))
        new_adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        nnz = adj.nnz
        perm = np.random.permutation(nnz)       #随机
        preserve_nnz = int(nnz*p)   #比例
        perm = perm[:preserve_nnz]  #删除
        r_adj = sp.coo_matrix((adj.data[perm], (adj.tocoo().row[perm], adj.tocoo().col[perm])), shape=adj.shape)
        r_adj = normalize(r_adj + sp.eye(r_adj.shape[0]))   #归一化
        new_adj = sparse_mx_to_torch_sparse_tensor(r_adj)

    return new_adj,new_adj.to_dense()

def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.
    """
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    if len(output.shape)==2:
        preds = output.max(1)[1].type_as(labels)
    else:
        preds=output
    f1 = f1_score(labels, preds, average='weighted')
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def task_generator(id_by_class, class_list, n_way, k_shot, m_query, maximum_value_train_each_class=None):

    # sample class indices
    class_selected = np.random.choice(class_list, n_way,replace=False).tolist()
    id_support = []
    id_query = []
    for cla in class_selected:
        if maximum_value_train_each_class:
            temp = np.random.choice(id_by_class[cla][:maximum_value_train_each_class], k_shot + m_query,replace=False)
        else:
            temp = np.random.choice(id_by_class[cla], k_shot + m_query,replace=False)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    return np.array(id_support), np.array(id_query), class_selected



def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M




valid_num_dic = {'Amazon_clothing': 17, 'Amazon_eletronics': 36, 'dblp': 27}
def load_data(dataset_source):
    class_list_train,class_list_valid,class_list_test=json.load(open('./dataset/{}_class_split.json'.format(dataset_source)))
    if dataset_source in valid_num_dic.keys():
        n1s = []
        n2s = []
        for line in open("dataset/{}_network".format(dataset_source)):
            n1, n2 = line.strip().split('\t')
            n1s.append(int(n1))
            n2s.append(int(n2))

        num_nodes = max(max(n1s),max(n2s)) + 1
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                    shape=(num_nodes, num_nodes))


        data_train = sio.loadmat("dataset/{}_train.mat".format(dataset_source))
        train_class = list(set(data_train["Label"].reshape((1,len(data_train["Label"])))[0]))
        

        data_test = sio.loadmat("dataset/{}_test.mat".format(dataset_source))
        class_list_test = list(set(data_test["Label"].reshape((1,len(data_test["Label"])))[0]))


        labels = np.zeros((num_nodes,1))
        labels[data_train['Index']] = data_train["Label"]
        labels[data_test['Index']] = data_test["Label"]

        features = np.zeros((num_nodes,data_train["Attributes"].shape[1]))
        features[data_train['Index']] = data_train["Attributes"].toarray()
        features[data_test['Index']] = data_test["Attributes"].toarray()

        class_list = []
        for cla in labels:
            if cla[0] not in class_list:
                class_list.append(cla[0])  # unsorted

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels):
            id_by_class[cla[0]].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)

        adj_cur = adj.copy()
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)
        
        #class_list_valid = random.sample(train_class, valid_num_dic[dataset_source])

        #class_list_train = list(set(train_class).difference(set(class_list_valid)))
    elif dataset_source=='cora-full':
        adj, features, labels, node_names, attr_names, class_names, metadata=load_npz_to_sparse_graph('./dataset/cora_full.npz')
             
        sparse_mx = adj.tocoo().astype(np.float32)
        indices =np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        
        n1s=indices[0].tolist()
        n2s=indices[1].tolist()
        
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        
        adj_cur = adj.copy()
        adj = normalize(adj.tocoo() + sp.eye(adj.shape[0]))
        adj= sparse_mx_to_torch_sparse_tensor(adj)
        num_nodes=adj.shape[0]
        print('nodes num',num_nodes)
        features=features.todense()
        features = torch.FloatTensor(features)
        labels=torch.LongTensor(labels).squeeze()
                
            
        class_list =  class_list_train+class_list_valid+class_list_test

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.numpy().tolist()):
            id_by_class[cla].append(id)
        
    elif dataset_source=='ogbn-arxiv':

        from ogb.nodeproppred import NodePropPredDataset

        dataset = NodePropPredDataset(name = dataset_source)
    
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, labels = dataset[0] # graph: library-agnostic graph object

        n1s=graph['edge_index'][0]
        n2s=graph['edge_index'][1]

        num_nodes = graph['num_nodes']
        print('nodes num',num_nodes)
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                shape=(num_nodes, num_nodes))    
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj_cur = adj.copy()

        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        features=torch.FloatTensor(graph['node_feat'])
        labels=torch.LongTensor(labels).squeeze()

        
        class_list =  class_list_train+class_list_valid+class_list_test

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.numpy().tolist()):
            id_by_class[cla].append(id)


    return adj_cur, adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class,num_nodes

torch.cuda.set_device(1)
print('gpu device:',1)

parser = argparse.ArgumentParser()

parser.add_argument('--use_cuda', action='store_true',default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')


parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of episodes to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--test_mode', type=str, default='GPN')



parser.add_argument('--way', type=int, default=10, help='way.')
parser.add_argument('--shot', type=int, default=3, help='shot.')
parser.add_argument('--qry', type=int, help='k shot for query set', default=10)
parser.add_argument('--data', default='dblp', help='Dataset:Amazon_clothing/Amazon_eletronics/dblp')
args = parser.parse_args()
args.cuda =  args.use_cuda


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# -------------------------Meta-training------------------------------


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


class GPN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GPN_Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return x


class GPN_Valuator(nn.Module):
    """
    For the sake of model efficiency, the current implementation is a little bit different from the original paper.
    Note that you can still try different architectures for building the valuator network.

    """

    def __init__(self, nfeat, nhid, dropout):
        super(GPN_Valuator, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.fc3 = nn.Linear(nhid, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.fc3(x)

        return x



def train (epoch,num_nodes,adj_sparse,class_selected, id_support, id_query, n_way, k_shot, cur=False):
    encoder.train()
    scorer.train()
    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    if cur==True:
            c = update_competency(epoch+1, args.epochs, 0.01, 2)
            #print('competency:',c)
            adj_sparse,adj = dropedge(adj=adj_sparse,num_nodes=num_nodes,p=c)

    #adj=adj.cuda()
    adj_sparse = adj_sparse.cuda()

    embeddings = encoder(features, adj_sparse)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj_sparse)

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # node importance
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot])+1e-9)
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_train = F.nll_loss(output, labels_new)

    loss_train.backward()
    optimizer_encoder.step()
    optimizer_scorer.step()

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)

    return acc_train, f1_train


def test(adj,class_selected, id_support, id_query, n_way, k_shot):
    encoder.eval()
    scorer.eval()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # node importance
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot])+1e-9)
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_test = F.nll_loss(output, labels_new)

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)

    return acc_test, f1_test


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler






n_query = args.qry
meta_test_num = 20
meta_valid_num = 20

# Sampling a pool of tasks for validation/testing

from collections import defaultdict

results=defaultdict(dict)


use_contrast=True
use_contrast_distinguish=False
use_label_supervise=True
use_contrast_normal=False
use_whether_label_contrast=True



use_predict_as_emb=False

save_time=datetime.datetime.now()

#names = ['Amazon_clothing', 'Amazon_eletronics', 'dblp']

#names=['ogbn-arxiv']
#names = ["cora-full"]
#for dataset in ['dblp','Amazon_clothing','Amazon_eletronics']:
for dataset in [args.data]:



    adj_cur, adj, features, labels, degrees, class_list_train, class_list_valid, class_list_test, id_by_class ,num_nodes = load_data(dataset)



    for n_way in [5,10]:
        for k_shot in [3,5]:
            for repeat in range(5):
                print('============================================',n_way,k_shot,repeat)
                encoder = GPN_Encoder(nfeat=features.shape[1], nhid=args.hidden, dropout=args.dropout)

                scorer = GPN_Valuator(nfeat=features.shape[1],
                                      nhid=args.hidden,
                                      dropout=args.dropout)

                optimizer_encoder = optim.Adam(encoder.parameters()
                                               , lr=args.lr, weight_decay=args.weight_decay)

                optimizer_scorer = optim.Adam(scorer.parameters(),
                                              lr=args.lr, weight_decay=args.weight_decay)

                if args.cuda:
                    encoder.cuda()
                    scorer.cuda()
                    features = features.cuda()
                   
                    labels = labels.cuda()
                    degrees = degrees.cuda()

                # Train model
                count=0
                best_valid_acc=0
                t_total = time.time()
                meta_train_acc = []
                for episode in range(args.epochs):

                    
                    id_support, id_query, class_selected = task_generator(id_by_class, class_list_train, n_way, k_shot, m_query=1,maximum_value_train_each_class=10)
                    
                    acc_train, f1_train = train(episode,num_nodes,adj,class_selected, id_support, id_query, n_way, k_shot)
                    #acc_train, f1_train = train(episode,num_nodes,adj_cur,class_selected, id_support, id_query, n_way, k_shot,cur=True)
                    meta_train_acc.append(acc_train)

                    if episode > 0 and episode % 10 == 0:
                        print("Meta-Train_Accuracy: {}".format(np.array(meta_train_acc).mean(axis=0)))


                        valid_pool = [task_generator(id_by_class, class_list_valid, n_way, k_shot, n_query) for i in range(meta_valid_num)]


                        # validation
                        meta_test_acc = []
                        meta_test_f1 = []
                        for idx in range(meta_valid_num):
                            id_support, id_query, class_selected = valid_pool[idx]

                            if args.test_mode!='LR':
                                adj=adj.cuda()
                                acc_test, f1_test = test(adj,class_selected, id_support, id_query, n_way, k_shot)
                            else:
                                acc_test, f1_test = LR_test(class_selected, id_support, id_query, n_way, k_shot)
                            meta_test_acc.append(acc_test)
                            meta_test_f1.append(f1_test)

                        valid_acc=np.array(meta_test_acc).mean(axis=0)
                        print("Meta-valid_Accuracy: {}, Meta-valid_F1: {},epoch: {}".format(valid_acc,
                                                                                  np.array(meta_test_f1).mean(axis=0),
                                                                                  episode))

                        if valid_acc>best_valid_acc:
                            best_valid_acc=valid_acc
                            count=0
                        else:
                            count+=1
                            if count>=10:
                                break

                # testing

                test_pool = [task_generator(id_by_class, class_list_test, n_way, k_shot, n_query) for i in range(meta_test_num)]
                meta_test_acc = []
                meta_test_f1 = []
                for idx in range(meta_test_num):
                    id_support, id_query, class_selected = test_pool[idx]

                    if args.test_mode!='LR':
                        adj=adj.cuda()
                        acc_test, f1_test = test(adj,class_selected, id_support, id_query, n_way, k_shot)
                    else:
                        acc_test, f1_test = LR_test(class_selected, id_support, id_query, n_way, k_shot)
                    meta_test_acc.append(acc_test)
                    meta_test_f1.append(f1_test)

                    if idx%20==0:
                        print("Task Num: {} Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(idx,np.array(meta_test_acc).mean(axis=0),
                                                                                np.array(meta_test_f1).mean(axis=0)))
                ori_accs=meta_test_acc.copy()
                #currcuilum
                
                count=0
                best_valid_acc=0
                t_total = time.time()
                for episode in range(args.epochs):

                    
                    id_support, id_query, class_selected = task_generator(id_by_class, class_list_train, n_way, k_shot, m_query=1,maximum_value_train_each_class=10)
                    
                    
                    # *****cur*****
                    acc_train, f1_train = train(episode,num_nodes,adj_cur,class_selected, id_support, id_query, n_way, k_shot,cur=True)
                    meta_train_acc.append(acc_train)

                    if episode > 0 and episode % 10 == 0:
                        print("Meta-Train_Accuracy: {}".format(np.array(meta_train_acc).mean(axis=0)))


                        valid_pool = [task_generator(id_by_class, class_list_valid, n_way, k_shot, n_query) for i in range(meta_valid_num)]


                        # validation
                        meta_test_acc = []
                        meta_test_f1 = []
                        for idx in range(meta_valid_num):
                            id_support, id_query, class_selected = valid_pool[idx]

                            if args.test_mode!='LR':
                                adj=adj.cuda()
                                acc_test, f1_test = test(adj,class_selected, id_support, id_query, n_way, k_shot)
                            else:
                                acc_test, f1_test = LR_test(class_selected, id_support, id_query, n_way, k_shot)
                            meta_test_acc.append(acc_test)
                            meta_test_f1.append(f1_test)

                        valid_acc=np.array(meta_test_acc).mean(axis=0)
                        print("Meta-valid_Accuracy: {}, Meta-valid_F1: {},epoch: {}".format(valid_acc,
                                                                                  np.array(meta_test_f1).mean(axis=0),
                                                                                  episode))

                        if valid_acc>best_valid_acc:
                            best_valid_acc=valid_acc
                            count=0
                        else:
                            count+=1
                            if count>=10:
                                break
                
                # testing

                test_pool = [task_generator(id_by_class, class_list_test, n_way, k_shot, n_query) for i in range(meta_test_num)]
                meta_test_acc = []
                #meta_test_f1 = []
                for idx in range(meta_test_num):
                    id_support, id_query, class_selected = test_pool[idx]

                    if args.test_mode!='LR':
                        adj=adj.cuda()
                        acc_test, f1_test = test(adj,class_selected, id_support, id_query, n_way, k_shot)
                    else:
                        acc_test, f1_test = LR_test(class_selected, id_support, id_query, n_way, k_shot)
                    meta_test_acc.append(acc_test)
                    #meta_test_f1.append(f1_test)

                    if idx%20==0:
                        print("Task Num: {} Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(idx,np.array(meta_test_acc).mean(axis=0),
                                                                                np.array(meta_test_f1).mean(axis=0)))



                results[dataset]['{}-way {}-shot {}-repeat'.format(n_way,k_shot,repeat)]=[np.array(meta_test_acc).mean(axis=0),
                                                            np.array(ori_accs).mean(axis=0)]
                json.dump(results,open('./2-GPN_result_{}.json'.format(dataset),'w'))

            accs=[]
            accs_ori=[]
            for repeat in range(5):
                accs.append(results[dataset]['{}-way {}-shot {}-repeat'.format(n_way,k_shot,repeat)][0])
                accs_ori.append(results[dataset]['{}-way {}-shot {}-repeat'.format(n_way,k_shot,repeat)][1])

            results[dataset]['{}-way {}-shot'.format(n_way,k_shot)]=[np.mean(accs),np.mean(accs_ori)]
            results[dataset]['{}-way {}-shot_print'.format(n_way,k_shot)]=['acc: {:.4f}'.format(np.mean(accs)),'ori_acc: {:.4f}'.format(np.mean(accs_ori))]


            json.dump(results,open('./2-GPN_result_{}.json'.format(dataset),'w'))


