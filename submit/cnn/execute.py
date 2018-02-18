import sys
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# How to: python3 execute.py train static

labels = ['neg','pos']

flag = sys.argv[1] # flag: train, test
flag2 = sys.argv[2] # flag2: static, tuned

pickle_file = "w2v_class1-a1_data.pickle"

# download data, vocabulary and embedding.

with open(pickle_file,'rb') as fp:
	save = pickle.load(fp)
	train_dataset = save['train_dataset']
	train_label = save['train_label']
	test_dataset = save['test_dataset']
	test_label = save['test_label']
	vocabulary = save['vocabulary']
	embeddings = save['embeddings']
	del save
	
filter_sizes_1 = 3
filter_sizes_2 = 4
filter_sizes_3 = 5

# l2 regularisation coefficient
l2_reg_lambda = 0.001

# num of training iterations
num_steps = 1001

# mini-batch size
batch_size = 50

# number of filters
num_filters = 100

# dropout keep probability
dropout_keep_prob = 0.5


if (flag2 == "static"):
	from cnn_static import TextCNN
if (flag2 == "tuned"):
	from cnn_tuned import TextCNN

if (flag == "train"):
	test_dataset = train_dataset
	test_label = train_label	
	cnn = TextCNN(train_dataset = train_dataset, train_labels = train_label, valid_dataset = test_dataset, valid_labels = test_label, embeddings = embeddings, vocabulary = vocabulary, l2_reg_lambda = l2_reg_lambda, num_steps = num_steps, batch_size = batch_size, num_filters = num_filters, filter_sizes_1 = filter_sizes_1, filter_sizes_2 = filter_sizes_2, filter_sizes_3 = filter_sizes_3, dropout_keep_prob = dropout_keep_prob)
	print ("valid accuracy: " +str(cnn.valid_accuracy))
	print ("\n")
	embeddings = cnn.embeddings_final
	print("="*10)

if (flag == "test"):
	cnn = TextCNN(train_dataset = train_dataset, train_labels = train_label, valid_dataset = test_dataset, valid_labels = test_label, embeddings = embeddings, vocabulary = vocabulary, l2_reg_lambda = l2_reg_lambda, num_steps = num_steps, batch_size = batch_size, num_filters = num_filters, filter_sizes_1 = filter_sizes_1, filter_sizes_2 = filter_sizes_2, filter_sizes_3 = filter_sizes_3, dropout_keep_prob = dropout_keep_prob)
	print ("valid accuracy: " +str(cnn.valid_accuracy))
	print ("\n")
	embeddings = cnn.embeddings_final
