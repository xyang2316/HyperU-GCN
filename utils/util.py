import pickle as pkl
import sys
import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from utils.normalization import fetch_normalization, row_normalize
from utils.npz_utils import load_npz_to_sparse_graph, to_binary_bag_of_words, eliminate_self_loops, binarize_labels
    
datadir = "data"
SEED = 42

    
def data_loader(dataset, data_path=datadir, normalization="AugNormAdj", porting_to_torch=True, task_type = "full"):
    if dataset in ["amazon_electronics_computers", "amazon_electronics_photo", "ms_academic_cs", "ms_academic_phy"]:
        (adj,
         features,
         labels,
         idx_train,
         idx_val,
         idx_test,
         degree,
         learning_type) = load_npz_data(dataset, SEED, task_type) 
        train_adj = adj
        train_features = features
        labels_1d = []
        for row in labels:
            for i, l in enumerate(row):
                if l:
                    labels_1d.append(i)
                    break
        labels = np.array(labels_1d)
        return adj, train_adj, features, train_features, labels, idx_train, idx_val, idx_test, degree, learning_type

    else:
        (adj,
         features,
         labels,
         idx_train,
         idx_val,
         idx_test,
         degree,
         learning_type) = load_citation(dataset, normalization, porting_to_torch, data_path, task_type)
        train_adj = adj
        train_features = features
        return adj, train_adj, features, train_features, labels, idx_train, idx_val, idx_test, degree, learning_type


def load_citation(dataset_str="citeseer", normalization="AugNormAdj", porting_to_torch=True, data_path=datadir, task_type="full"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str.lower(), names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects) 
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) 
    degree = np.sum(adj, axis=1)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    if task_type == "full":
        print("Load full supervised task.")
        idx_test = test_idx_range.tolist()
        idx_train = range(len(ally)- 500)
        idx_val = range(len(ally) - 500, len(ally))
    elif task_type == "semi":
        print("Load semi-supervised task.")
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)
    else:
        raise ValueError("Task type: %s is not supported. Available option: full and semi.")
    
    # create adj for cnn and normalize features
    adj, features = preprocess_citation(adj, features, normalization)
    features = np.array(features.todense())
    labels = np.argmax(labels, axis=1)
    # porting to pytorch
    if porting_to_torch:
        features = torch.FloatTensor(features).float()
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        degree = torch.LongTensor(degree)
    learning_type = "transductive"
    return adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type


def load_npz_data(dataset_str, seed, task_type): 
    adj, features, labels = get_dataset("data/{}.npz".format(dataset_str), True)
    features = features.toarray() 
    adj = adj.tocoo()
    random_state = np.random.RandomState(seed)
    if task_type == "semi":
        y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_size=20*labels.shape[1], val_size=30*labels.shape[1]) 
    if task_type == "full":
        train_size = labels.shape[0] - 30*labels.shape[1] - 1000
        y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_size=train_size, val_size=30*labels.shape[1], test_size=1000) 
    degree = None
    learning_type = None
    return adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type


def data_loader_OOD(dataset, data_path=datadir, normalization="AugNormAdj", porting_to_torch=True, task_type = "full"):
    if dataset in ["cora", "citeseer", "pubmed"]:
        (adj,
         features,
         labels,
         idx_train,
         idx_val,
         idx_test,
         degree,
         learning_type,
         idx_test_ood,
         idx_test_id) = load_citation_ood(dataset, normalization, porting_to_torch, data_path, task_type)
        print("citation", labels)
        train_adj = adj
        train_features = features
        return adj, train_adj, features, train_features, labels, idx_train, idx_val, idx_test, degree, learning_type, idx_test_ood, idx_test_id
    elif dataset in ["amazon_electronics_computers", "amazon_electronics_photo", "ms_academic_cs", "ms_academic_phy"]:
        (adj,
         features,
         labels,
         idx_train,
         idx_val,
         idx_test,
         degree,
         learning_type,
         idx_test_ood,
         idx_test_id) = load_npz_data_ood_train(dataset, SEED) 
        train_adj = adj
        train_features = features
        return adj, train_adj, features, train_features, labels, idx_train, idx_val, idx_test, degree, learning_type, idx_test_ood, idx_test_id

    else:
        print("dataset is NOT available for OOD") 


def load_citation_ood(dataset_str="citeseer", normalization="AugNormAdj", porting_to_torch=True, data_path=datadir, task_type="full"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str.lower(), names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects) 
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)  

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) 
    degree = np.sum(adj, axis=1)
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if task_type == "full":
        print("Load full supervised task.")
        idx_test = test_idx_range.tolist()
        idx_train = range(len(ally)- 500)
        idx_val = range(len(ally) - 500, len(ally))
    elif task_type == "semi":
        print("Load semi-supervised task.")
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)
    else:
        raise ValueError("Task type: %s is not supported. Available option: full and semi.")
    
    # create adj for cnn and normalize features
    adj, features = preprocess_citation(adj, features, normalization)
    features = np.array(features.todense())

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    test_mask_ood = sample_mask(idx_test, labels.shape[0])
    test_mask_id = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    category = np.argmax(labels, axis=1)
    labels = np.argmax(labels, axis=1)
    
    if dataset_str == 'cora':
        for i in range(labels.shape[0]):  
            #OOD
            if category[i] > 3:
                train_mask[i] = False  
                val_mask[i] = False
                test_mask_id[i] = False
                labels[i] = -1
            #ID
            else:
                test_mask_ood[i] = False  

    if dataset_str == 'citeseer':
        for i in range(labels.shape[0]):
            if category[i] > 2:
                train_mask[i] = False
                val_mask[i] = False
                test_mask_id[i] = False
                labels[i] = -1
            else:
                test_mask_ood[i] = False

    if dataset_str == 'pubmed':
        for i in range(labels.shape[0]):
            if category[i] > 1:
                train_mask[i] = False
                val_mask[i] = False
                test_mask_id[i] = False
                labels[i] = -1
            else:
                test_mask_ood[i] = False

    idx_train = [i for i, ele in enumerate(train_mask) if ele == True]
    idx_val = [i for i, ele in enumerate(val_mask) if ele == True]
    idx_test = [i for i, ele in enumerate(test_mask) if ele == True]
    idx_test_ood = [i for i, ele in enumerate(test_mask_ood) if ele == True]
    idx_test_id = [i for i, ele in enumerate(test_mask_id) if ele == True]
    learning_type = "transductive" 
    return adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type, idx_test_ood, idx_test_id


def load_npz_data_ood_train(dataset_str, seed):  
    adj, features, labels = get_dataset("data/{}.npz".format(dataset_str), True)
    features = features.toarray() 
    adj = adj.tocoo()
    
    random_state = np.random.RandomState(seed)
    y_train, y_val, y_test, train_mask, val_mask, test_mask, _ = get_train_val_test_split_ood(random_state, labels, train_size=20*labels.shape[1], val_size=30*labels.shape[1])
    
    test_mask = np.array(1 - train_mask - val_mask, dtype=bool)
    category = np.argmax(labels, axis=1)
    test_mask_all = np.array(test_mask)
    test_mask_id = np.array(test_mask)
    test_mask_ood = np.array(test_mask)

    idcount = 0
    y_train = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]

    labels = np.argmax(labels, axis=1)
    if dataset_str == 'amazon_electronics_photo':
        for i in range(labels.shape[0]):
            if category[i] > 3:
                train_mask[i] = False
                val_mask[i] = False
                test_mask_id[i] = False
                labels[i] = -1
            else:
                test_mask_ood[i] = False
                idcount += 1
    if dataset_str == 'amazon_electronics_computers':
        for i in range(labels.shape[0]):
            if category[i] > 4:
                train_mask[i] = False
                val_mask[i] = False
         
                test_mask_id[i] = False
                labels[i] = -1
            else:
                test_mask_ood[i] = False
    if dataset_str == 'ms_academic_phy':
        for i in range(labels.shape[0]):
            if category[i] > 2:
                train_mask[i] = False
                val_mask[i] = False
                test_mask_id[i] = False
                labels[i] = -1
            else:
        
                test_mask_ood[i] = False
    degree = None
    learning_type = None
    idx_train = np.array([i for i, ele in enumerate(train_mask) if ele == True])
    idx_val = np.array([i for i, ele in enumerate(val_mask) if ele == True])
    idx_test = np.array([i for i, ele in enumerate(test_mask) if ele == True])
    idx_test_ood = np.array([i for i, ele in enumerate(test_mask_ood) if ele == True])
    idx_test_id = np.array([i for i, ele in enumerate(test_mask_id) if ele == True])
    return adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type, idx_test_ood, idx_test_id
  

def get_train_val_test_split_ood(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)
        train_mask = sample_mask(train_indices, labels.shape[0])

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)
        val_mask = sample_mask(val_indices, labels.shape[0])

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
        test_mask = sample_mask(test_indices, labels.shape[0])
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_mask = sample_mask(test_indices, labels.shape[0])

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask, train_indices

def compute_entropy(u_out, index, c):
    probs = u_out[index]
    mean = []
    for p in probs:
        mean.append(p.detach().cpu().numpy())
    class_num = c
    prob = np.array(mean)
    entropy = - prob * (np.log(prob) / np.log(class_num))
    class_un = np.mean(entropy, axis=0) 
    total_un = np.sum(class_un)
    return total_un, class_un

def get_dataset(data_path, standardize):
    dataset_graph = load_npz_to_sparse_graph(data_path)

    if standardize:
        dataset_graph = dataset_graph.standardize()
    else:
        dataset_graph = dataset_graph.to_undirected()
        dataset_graph = eliminate_self_loops(dataset_graph)

    adj_matrix, attr_matrix, labels = dataset_graph.unpack()

    labels = binarize_labels(labels)
    # convert to binary bag-of-words feature representation if necessary
    if not is_binary_bag_of_words(attr_matrix):
        attr_matrix = to_binary_bag_of_words(attr_matrix)

    # some assertions that need to hold for all datasets
    # adj matrix needs to be symmetric
    # assert (adj_matrix != adj_matrix.T).nnz == 0
    # features need to be binary bag-of-word vectors
    # assert is_binary_bag_of_words(attr_matrix), f"Non-binary node_features entry!"

    return adj_matrix, attr_matrix, labels

def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)
        train_mask = sample_mask(train_indices, labels.shape[0])

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)
        val_mask = sample_mask(val_indices, labels.shape[0])

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
        test_mask = sample_mask(test_indices, labels.shape[0])
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_mask = sample_mask(test_indices, labels.shape[0])

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
   
    idx_train = train_indices
    idx_val = val_indices
    idx_test = test_indices
   
    return y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, idx_val, idx_test

def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])

def is_binary_bag_of_words(features):
    features_coo = features.tocoo()
    return all(single_entry == 1.0 for _, _, single_entry in zip(features_coo.row, features_coo.col, features_coo.data))

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def convert2one_hot(labels, nclass, device):
    labels = labels.tolist()
    one_hot = []
    for l in range(len(labels)):
        cur_sample = []
        for i in range(nclass):
            if i == labels[l]:
                cur_sample.append(1)
            else:
                cur_sample.append(0)
        one_hot.append(cur_sample)
    return torch.Tensor(one_hot).to(device)

