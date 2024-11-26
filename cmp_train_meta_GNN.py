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
import time
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score
# import contrast_util
import json
import os
import time
import gc
# import GCL.losses as L
# import GCL.augmentors as A
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# from GCL.eval import get_split, LREvaluator
# from GCL.models import DualBranchContrast
from base_model_SSL import GCN_dense
from base_model_SSL import Linear
from base_model_SSL import GCN_emb
import higher



# from base_model import GCN

def update_competency(t, T, c_0, p):
    term = pow(((1 - pow(c_0,p))*(t/T)) + pow(c_0,p), (1/p))
    return min([1,term])

def l2_normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm+1e-10)
    return out

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
    preds = output.max(1)[1].type_as(labels)
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


def cal_euclidean(input):
    # input tensor
    #a = input.unsqueeze(0).repeat([input.shape[0], 1, 1])
    #b = input.unsqueeze(1).repeat([1, input.shape[0], 1])
    #distance = (a - b).square().sum(-1)
    distance = torch.cdist(input.unsqueeze(0),input.unsqueeze(0)).squeeze()
    
    return distance


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

def dgl_dropedge(adj,num_nodes,p=0.5,drop=True):


    p=1-p
    if drop==False:

        adj = normalize(adj + sp.eye(adj.shape[0]))
        new_adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:

        nnz = adj.nnz
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz*p)
        perm = perm[:preserve_nnz]
        r_adj = sp.coo_matrix((adj.data[perm], (adj.tocoo().row[perm], adj.tocoo().col[perm])), shape=adj.shape)
        r_adj = normalize(r_adj + sp.eye(r_adj.shape[0]))
        new_adj = sparse_mx_to_torch_sparse_tensor(r_adj)
    

    return new_adj,new_adj.to_dense()


valid_num_dic = {'Amazon_eletronics': 36, 'dblp': 27}

def load_data_pretrain(dataset_source):
    class_list_train,class_list_valid,class_list_test=json.load(open('./dataset/{}_class_split.json'.format(dataset_source)))
    if dataset_source in valid_num_dic.keys():

        n1s = []
        n2s = []
        for line in open("./dataset/{}_network".format(dataset_source)):
            n1, n2 = line.strip().split('\t')
            n1s.append(int(n1))
            n2s.append(int(n2))

        data_train = sio.loadmat("./dataset/{}_train.mat".format(dataset_source))
        data_test = sio.loadmat("./dataset/{}_test.mat".format(dataset_source))

        num_nodes = max(max(n1s),max(n2s)) + 1
        labels = np.zeros((num_nodes,1))
        labels[data_train['Index']] = data_train["Label"]
        labels[data_test['Index']] = data_test["Label"]

        features = np.zeros((num_nodes,data_train["Attributes"].shape[1]))
        features[data_train['Index']] = data_train["Attributes"].toarray()
        features[data_test['Index']] = data_test["Attributes"].toarray()


        print('nodes num',num_nodes)
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                shape=(num_nodes, num_nodes))    

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

        adj_cur = adj.copy()
        adj = normalize(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)

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


    idx_train,idx_valid,idx_test=[],[],[]
    for idx_,class_list_ in zip([idx_train,idx_valid,idx_test],[class_list_train,class_list_valid,class_list_test]):
        for class_ in class_list_:
            idx_.extend(id_by_class[class_])

    class_train_dict=defaultdict(list)
    for one in class_list_train:
        for i,label in enumerate(labels.numpy().tolist()):
            if label==one:
                class_train_dict[one].append(i)
    class_valid_dict = defaultdict(list)
    for one in class_list_valid:
        for i, label in enumerate(labels.numpy().tolist()):
            if label == one:
                class_valid_dict[one].append(i)


    class_test_dict = defaultdict(list)
    for one in class_list_test:
        for i, label in enumerate(labels.numpy().tolist()):
            if label == one:
                class_test_dict[one].append(i)
    return adj_cur,adj, features, labels, idx_train, idx_valid, idx_test, n1s, n2s, class_train_dict, class_test_dict, class_valid_dict



def neighborhoods_(adj, n_hops, use_cuda):
    """Returns the n_hops degree adjacency matrix adj."""
    # adj = torch.tensor(adj, dtype=torch.float)
    # adj=adj.to_dense()
    # print(type(adj))
    if use_cuda:
        adj = adj.cuda()
    # hop_adj = power_adj = adj

    # return (adj@(adj.to_dense())+adj).to_dense().cpu().numpy().astype(int)

    hop_adj = adj + torch.sparse.mm(adj, adj)

    hop_adj = hop_adj.to_dense()
    # hop_adj = (hop_adj > 0).to_dense()

    # for i in range(n_hops - 1):
    # power_adj = power_adj @ adj
    # prev_hop_adj = hop_adj
    # hop_adj = hop_adj + power_adj
    # hop_adj = (hop_adj > 0).float()

    hop_adj = hop_adj.cpu().numpy().astype(int)

    return (hop_adj > 0).astype(int)

    # return hop_adj.cpu().numpy().astype(int)


def neighborhoods(adj, n_hops, use_cuda):
    """Returns the n_hops degree adjacency matrix adj."""
    # adj = torch.tensor(adj, dtype=torch.float)
    # adj=adj.to_dense()
    # print(type(adj))
    if n_hops == 1:
        return adj.cpu().numpy().astype(int)

    if use_cuda:
        adj = adj.cuda()
    # hop_adj = power_adj = adj

    # for i in range(n_hops - 1):
    # power_adj = power_adj @ adj
    hop_adj = adj + adj @ adj
    hop_adj = (hop_adj > 0).float()

    np.save(hop_adj.cpu().numpy().astype(int), './neighborhoods_{}.npy'.format(dataset))

    return hop_adj.cpu().numpy().astype(int)


def InforNCE_Loss(anchor, sample, tau, all_negative=False, temperature_matrix=None):
    def _similarity(h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()

    assert anchor.shape[0] == sample.shape[0]

    
    pos_mask = torch.eye(anchor.shape[0], dtype=torch.float)
    if dataset!='ogbn-arxiv':
        pos_mask=pos_mask.cuda()
    
    neg_mask = 1. - pos_mask

    sim = _similarity(anchor, sample / temperature_matrix if temperature_matrix != None else sample) / tau
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)

    if not all_negative:
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    else:
        log_prob = - torch.log(exp_sim.sum(dim=1, keepdim=True))

    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)

    return -loss.mean(), sim

class Predictor(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout=0):
        super(Predictor, self).__init__()
        self.linear1 = Linear(nfeat, nhid)
        self.linear2 = Linear(nhid, nout)

    def forward(self, x):
        return self.linear2(self.linear1(x).relu())


parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--test_epochs', type=int, default=20,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')

parser.add_argument('--weight_decay', type=float, default=5e-4,  # 5e-4
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--data', type=str, default='cora-full')
parser.add_argument('--name', type=str, default='ori')
parser.add_argument('--gpu', type=int, default=4)

args = parser.parse_args()
torch.cuda.set_device(args.gpu)


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.use_cuda:
    torch.cuda.manual_seed(args.seed)

loss_f = nn.CrossEntropyLoss()


Q=10


fine_tune_steps = 5
fine_tune_lr = 0.05


results=defaultdict(dict)

import sys
#for dataset in ['cora-full','dblp','Amazon_eletronics','ogbn-arxiv']:
for dataset in [args.data]:

    adj_sparse_cur, adj_sparse, features, labels, idx_train, idx_val, idx_test, n1s, n2s, class_train_dict, class_test_dict, class_valid_dict = load_data_pretrain(
        dataset)


    if dataset!='ogbn-arxiv':
        args.use_cuda=True
    else:
        args.use_cuda=False


    N_set=[5,10]
    K_set=[3,5,10]


    for N in N_set:
        for K in K_set:
            for repeat in range(5):

                adj = adj_sparse.to_dense()
                print('done')
                print(dataset)
                print('N={},K={},repeat={}'.format(N,K,repeat))


                model = GCN_dense(nfeat=features.shape[1],
                                  nhid=args.hidden2,
                                  nclass=labels.max().item() + 1,
                                  dropout=args.dropout)




                classifier = Linear(args.hidden1, labels.max().item() + 1)
                predictor = Predictor(args.hidden2, args.hidden2 * 2, args.hidden2)
                optimizer = optim.Adam([{'params': model.parameters()}, {'params': classifier.parameters()},{'params': predictor.parameters()}],
                                       lr=args.lr, weight_decay=args.weight_decay)

                if args.use_cuda:
                    model.cuda()
                    features = features.cuda()
                    predictor=predictor.cuda()
                    labels = labels.cuda()
                    classifier = classifier.cuda()


                def pre_train(epoch,adj, adj_sparse, N, mode='train',cur=False):

                    support_labels=torch.zeros(N*K,dtype=torch.long)
                    for i in range(N):
                        support_labels[i * K:(i + 1) * K] = i
                    query_labels=torch.zeros(N*Q,dtype=torch.long)
                    for i in range(N):
                        query_labels[i * Q:(i + 1) * Q] = i
                    if args.use_cuda:
                        support_labels=support_labels.cuda()
                        query_labels=query_labels.cuda()


                    if mode == 'train':
                        model.train()
                        optimizer.zero_grad()
                    else:
                        model.eval()


                    if cur==True: 
                        c = update_competency(epoch+1, args.epochs, 0.01, 2)
                        adj_sparse,adj = dgl_dropedge(adj=adj_sparse,num_nodes=adj.shape[0],p=c)
                    if args.use_cuda:
                        adj=adj.cuda()
                        adj_sparse = adj_sparse.cuda()



                    target_idx = []
                    target_new_idx = []
                    target_graph_adj_and_feat = []
                    support_target_graph_adj_and_feat = []

                    pos_node_idx = []

                    if mode == 'train':
                        class_dict = class_train_dict
                    elif mode == 'test':
                        class_dict = class_test_dict
                    elif mode=='valid':
                        class_dict = class_valid_dict
                    
                    classes = np.random.choice(list(class_dict.keys()), N, replace=False).tolist()

                    for i in classes:
                        # sample from one specific class
                        sampled_idx=np.random.choice(class_dict[i], K+Q, replace=False).tolist()
                        pos_node_idx.extend(sampled_idx[:K])
                        target_idx.extend(sampled_idx[K:])

                    #gc1_w, gc1_b, gc2_w, gc2_b, w, b = model.gc1.weight, model.gc1.bias, model.gc2.weight, model.gc2.bias, classifier.weight, classifier.bias


                    #print_memory_usage("Before higher.innerloop_ctx")
                    opt_fintune = torch.optim.Adam([{'params': model.parameters()}], lr=fine_tune_lr)
                    with higher.innerloop_ctx(model, opt_fintune, copy_initial_weights=True) as (fmodel, diffopt):
                        for j in range(fine_tune_steps):
                            #opt_fintune.zero_grad()  

                            emb_features = fmodel(features, adj_sparse)
                            ori_emb = emb_features[pos_node_idx]

                            loss_supervised = loss_f(classifier(ori_emb), support_labels)


                            #grad = torch.autograd.grad(loss_supervised, [gc1_w, gc1_b, gc2_w, gc2_b, w, b])
                            #gc1_w, gc1_b, gc2_w, gc2_b, w, b = list(
                                #map(lambda p: p[1] - fine_tune_lr * p[0], zip(grad, [gc1_w, gc1_b, gc2_w, gc2_b, w, b])))

                            diffopt.step(loss_supervised)

                        

                        fmodel.eval()

                        emb_features = fmodel(features, adj_sparse)
                        ori_emb = emb_features[target_idx]
                        logits = classifier(ori_emb)
                        query_loss  = loss_f(logits, query_labels)


                        if mode == 'train':
                            query_loss.backward()
                            optimizer.step()

                    #print_memory_usage("After higher.innerloop_ctx")
                    del fmodel, diffopt
                    gc.collect()
                    torch.cuda.empty_cache()

                    if epoch % 200 == 0 and mode == 'train':
                        print('Epoch: {:04d}'.format(epoch + 1),
                            'loss_train: {:.4f}'.format(query_loss.item()),
                            'acc_train: {:.4f}'.format((torch.argmax(logits, -1) == query_labels).float().mean().item()))
                    return (torch.argmax(logits, -1) == query_labels).float().mean().item()




                # Train model
                #adj_cur=adj_sparse_cur.todense()
                t_total = time.time()
                best_acc = 0
                best_valid_acc=0
                count=0
                for epoch in range(args.epochs):
                    t=time.time()
                    acc_train=pre_train(epoch,adj,adj_sparse, N=N,cur=False)
                    
	           
                    if  epoch > 0 and epoch % 50 == 0:

                        temp_accs=[]
                        for epoch_test in range(20):
                            temp_accs.append(pre_train(epoch_test,adj,adj_sparse, N=N, mode='test'))

                        accs = []

                        for epoch_test in range(20):
                            accs.append(pre_train(epoch_test,adj,adj_sparse, N=N if dataset!='ogbn-arxiv' else 5, mode='valid'))

                        valid_acc=np.array(accs).mean(axis=0)
                        print("Epoch: {:04d} Meta-valid_Accuracy: {:.4f}".format(epoch + 1, valid_acc))


                        if valid_acc>best_valid_acc:
                            best_test_accs=temp_accs
                            best_valid_acc=valid_acc
                            count=0

                        else:
                            count+=1
                            if count>=10:       #早停轮数10
                                break
                    #print(time.time()-t)   
                ori_accs=best_test_accs.copy()

                # curriculum
                adj_cur=adj_sparse_cur.todense()
                t_total = time.time()
                best_acc = 0
                best_valid_acc=0
                count=0
                for epoch in range(args.epochs):
                    t=time.time()
                    acc_train=pre_train(epoch,adj_cur,adj_sparse_cur, N=N,cur=True)
                    
	           
                    if  epoch > 0 and epoch % 50 == 0:

                        temp_accs=[]
                        for epoch_test in range(20):
                            temp_accs.append(pre_train(epoch_test,adj,adj_sparse, N=N, mode='test'))

                        accs = []

                        for epoch_test in range(20):
                            accs.append(pre_train(epoch_test,adj,adj_sparse, N=N if dataset!='ogbn-arxiv' else 5, mode='valid'))

                        valid_acc=np.array(accs).mean(axis=0)
                        print("Epoch: {:04d} Meta-valid_Accuracy: {:.4f}".format(epoch + 1, valid_acc))


                        if valid_acc>best_valid_acc:
                            best_test_accs=temp_accs
                            best_valid_acc=valid_acc
                            count=0

                        else:
                            count+=1
                            if count>=5:       #早停轮数10
                                break
                    #print(time.time()-t)    

                accs=best_test_accs

                print('Test Acc',np.array(accs).mean(axis=0))
                results[dataset]['{}-way {}-shot {}-repeat'.format(N,K,repeat)]=[np.array(accs).mean(axis=0),np.array(ori_accs).mean(axis=0) ]


                json.dump(results[dataset],open('./rebuttal-Meta-GNN-result_{}.json'.format(dataset),'w'))
                # 删除模型和相关变量，释放显存
                del model, classifier, predictor, optimizer
                del  adj,adj_cur
                torch.cuda.empty_cache()  # 强制释放显存


            accs=[]
            accs_ori=[]

            for repeat in range(5):
                accs.append(results[dataset]['{}-way {}-shot {}-repeat'.format(N,K,repeat)][0])
                accs_ori.append(results[dataset]['{}-way {}-shot {}-repeat'.format(N,K,repeat)][1])



            results[dataset]['{}-way {}-shot'.format(N,K)]=[np.mean(accs),np.mean(accs_ori)]
            results[dataset]['{}-way {}-shot_print'.format(N,K)]=['acc: {:.4f}'.format(np.mean(accs)),'ori_acc: {:.4f}'.format(np.mean(accs_ori))]


            json.dump(results[dataset],open('./rebuttal-Meta-GNN-result_{}.json'.format(dataset),'w'))   

            del model

    #del adj
