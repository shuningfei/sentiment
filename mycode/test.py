import sys
import pickle
import numpy as np

pickle_file = 'w2v_class1c-a1_data.pickle'

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

l2_reg_lambda = 0.001
num_steps = 1001
batch_size = 50
num_filters = 100
dropout_keep_prob = 0.5

vocab_size = len(vocabulary)
sequence_length = train_dataset.shape[1]
train_size = train_dataset.shape[0]
num_classes = 2

filter_sizes = [filter_sizes_1, filter_sizes_2, filter_sizes_3]
num_filters_total = num_filters * len(filter_sizes)
embedding_size = embeddings.shape[1]
embeddings_number = embeddings.shape[0]

print (embedding_size)
print (embeddings_number)