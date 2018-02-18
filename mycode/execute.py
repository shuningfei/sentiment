import sys
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# How to: python3 execute.py train static

labels = ['neg','pos']

flag = sys.argv[1] # flag: train, test
flag2 = sys.argv[2] # flag2: static, tuned

vectors = "w2v"

pickle_file = vectors + '_class3subj-a1_data.pickle'

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


if (flag2 == "static"):
	from cnn_class_fm_static import TextCNN
if (flag2 == "tuned"):
	from cnn_class_fm_tuned import TextCNN

if (flag == "train"):
	test_dataset = train_dataset
	test_label = train_label	
	cnn = TextCNN(train_dataset = train_dataset, train_labels = train_label, valid_dataset = test_dataset, valid_labels = test_label, embeddings = embeddings, vocabulary = vocabulary, l2_reg_lambda = l2_reg_lambda, num_steps = num_steps, batch_size = batch_size, num_filters = num_filters, filter_sizes_1 = filter_sizes_1, filter_sizes_2 = filter_sizes_2, filter_sizes_3 = filter_sizes_3, dropout_keep_prob = dropout_keep_prob, lexical = "lex", shuffling = "n")
	print (cnn.valid_accuracy)
	print ("\n")
	embeddings = cnn.embeddings_final
	print("="*10)

if (flag == "test"):
	cnn = TextCNN(train_dataset = train_dataset, train_labels = train_label, valid_dataset = test_dataset, valid_labels = test_label, embeddings = embeddings, vocabulary = vocabulary, l2_reg_lambda = l2_reg_lambda, num_steps = num_steps, batch_size = batch_size, num_filters = num_filters, filter_sizes_1 = filter_sizes_1, filter_sizes_2 = filter_sizes_2, filter_sizes_3 = filter_sizes_3, dropout_keep_prob = dropout_keep_prob, lexical= "lex", shuffling = "n")
	print (cnn.valid_accuracy)
	print ("\n")
	embeddings = cnn.embeddings_final

vocabulary_inv = [None] * len(vocabulary)

for word in vocabulary:
	vocabulary_inv[vocabulary[word]] = word

argmax1 = np.load("argmax_filter_sizes_1_"+flag2+".npy")
argmax2 = np.load("argmax_filter_sizes_2_"+flag2+".npy")
argmax3 = np.load("argmax_filter_sizes_3_"+flag2+".npy")

argmax1 = argmax1[0,:,0,:]
argmax2 = argmax2[0,:,0,:]
argmax3 = argmax3[0,:,0,:]

top_num = 15

#print (argmax1.shape)
data_num = argmax1.shape[0]

#print (data_num)

# 4659,100 -> 4659 is traning size, 100 is 100 filters

# 100 is filter_size

argmax_sent_1 = np.zeros((data_num,100))
argmax_sent_2 = np.zeros((data_num,100))
argmax_sent_3 = np.zeros((data_num,100))

maximum1 = np.load("maximum_filter_sizes_1_"+flag2+".npy")
maximum2 = np.load("maximum_filter_sizes_2_"+flag2+".npy")
maximum3 = np.load("maximum_filter_sizes_3_"+flag2+".npy")

maximum1 = maximum1[0,:,0,:]
maximum2 = maximum2[0,:,0,:]
maximum3 = maximum3[0,:,0,:]

print (maximum1.shape)

# 4659,100

"""
for i in range(100):
	argmax_sent_1[:,i] = maximum1[:,i].argsort()[-top_num:][::-1]
	argmax_sent_2[:,i] = maximum2[:,i].argsort()[-top_num:][::-1]
	argmax_sent_3[:,i] = maximum3[:,i].argsort()[-top_num:][::-1]
"""

for i in range(100):
	argmax_sent_1[:,i] = np.array(range(maximum1[:,i].shape[0]))
	argmax_sent_2[:,i] = np.array(range(maximum2[:,i].shape[0]))
	argmax_sent_3[:,i] = np.array(range(maximum3[:,i].shape[0]))

	
argmax_sent_1 = argmax_sent_1.astype(int)
argmax_sent_2 = argmax_sent_2.astype(int)
argmax_sent_3 = argmax_sent_3.astype(int)

labels = ['neg', 'pos']

prediction_labels = np.load("gold_labels_"+flag2+".npy")

for i in range(argmax1.shape[1]):
	f = open("/proj/zhou/sentiment/mycode/feature_maps_"+ flag + "_" + flag2 + "/fm_filter_sizes_1_"+str(i+1), "w")
	argmaxs = argmax1[:,i]
	
	#labels_stats = [" "]*top_num
	
	for k in range(argmax_sent_1.shape[0]):
		j = argmax_sent_1[k,i]

		f.write(labels[np.argmax(test_label[j,:])]) 
		f.write("\t\t")
		f.write(labels[prediction_labels[j]])
		f.write("\t\t")
		
		#labels_stats[k] = labels[prediction_labels[j]]
		
		argmax = int(argmaxs[j])
		
		ngram = test_dataset[j,argmax:argmax+filter_sizes_1]
		
		pad = vocabulary["<PAD/>"]

		for word in ngram:
			if(word != pad):
				f.write(str(vocabulary_inv[word]) + " ")
		f.write("\t\t")
		
		for word in test_dataset[j,:]:
			if(word != pad):
				f.write(str(vocabulary_inv[word]) + " ")
		f.write("\n")
	