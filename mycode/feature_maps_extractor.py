import sys
import pickle
import numpy as np
from cnn_class_fm_static import TextCNN

labels = ['pos','neg']

flag = sys.argv[1]

vectors = "w2v"
tuning = "static" 	#??
lexical = "lex"		#??

pickle_file = vectors + '_data.pickle'

with open(pickle_file, 'rb') as fp:
	save = pickle.load(fp)
	train_dataset = save['train_dataset']
	train_label = save['train_label']
	test_dataset = save['test_dataset']
	test_label = save['test_label']
	vocabulary = save['vocabulary']
	embeddings = save['embeddings']

filter_sizes_1 = 3
filter_sizes_2 = 4
filter_sizes_3 = 5

l2_reg_lambda = 0.001 # L2 regularization ??
num_steps = 1001 #??
batch_size = 50
num_filters = 100
dropout_keep_prob = 0.5


if (flag == "train"):
	test_dataset = train_dataset
	test_label = train_label
	cnn = TextCNN(train_dataset = train_dataset, train_labels = train_label, valid_dataset = test_dataset, valid_labels = test_label, embeddings = embeddings, vocabulary = vocabulary, l2_reg_lambda = l2_reg_lambda, num_steps = num_steps, batch_size = batch_size, num_filters = num_filters, filter_sizes_1 = filter_sizes_1, filter_sizes_2 = filter_sizes_2, filter_sizes_3 = filter_sizes_3, dropout_keep_prob = dropout_keep_prob, lexical = "lex", shuffling = "n")
	print (cnn.valid_accuracy)
	print ("\n")
	embeddings = cnn.embeddings_final
	#print(embeddings)
	print("="*10)

if (flag == "test"):
	cnn = TextCNN(train_dataset = train_dataset, train_labels = train_label, valid_dataset = test_dataset, valid_labels = test_label, embeddings = embeddings, vocabulary = vocabulary, l2_reg_lambda = l2_reg_lambda, num_steps = num_steps, batch_size = batch_size, num_filters = num_filters, filter_sizes_1 = filter_sizes_1, filter_sizes_2 = filter_sizes_2, filter_sizes_3 = filter_sizes_3, dropout_keep_prob = dropout_keep_prob, lexical= "lex", shuffling = "n")
	print (cnn.valid_accuracy)
	print ("\n")
	embeddings = cnn.embeddings_final
	#print(embeddings)
	print("="*10)

vocabulary_inv = [None] * len(vocabulary)
print (vocabulary_inv)
# [None,None,None,None, ... ]

for word in vocabulary:
	vocabulary_inv[vocabulary[word]] = word 
print (vocabulary_inv)

argmax1 = np.load("argmax_filter_sizes_1.npy")
argmax2 = np.load("argmax_filter_sizes_2.npy")
argmax3 = np.load("argmax_filter_sizes_3.npy")

#print ("argmax1"+str(argmax1))
#print ("argmax2"+str(argmax2))
#print ("argmax3"+str(argmax3))
	
"""
argmax1[[[[ 5  3  0 ...,  2  3  4]]

  [[10  9  1 ...,  5 12  0]]

  [[11 13  6 ..., 19  9  9]]

  ..., 
  [[ 9 13  4 ..., 14  6  5]]

  [[ 8 17  9 ...,  4 38 12]]

  [[ 2  0  0 ...,  4  0  1]]]]
argmax2[[[[ 2  1  2 ...,  2  1  1]]

  [[ 3  2  0 ...,  0  6 12]]

  [[15 11 13 ..., 12 11 11]]

  ..., 
  [[13  6  3 ..., 14  6  6]]

  [[ 3 36 25 ..., 18 10 39]]

  [[ 2  1  2 ...,  2  1  0]]]]
argmax3[[[[ 5  0  2 ...,  0  5  0]]

  [[ 0  0  4 ...,  5 13  3]]

  [[ 9  4  3 ..., 12  8  8]]

  ..., 
  [[12 12  0 ...,  1  7  6]]

  [[ 8 13 25 ..., 16 16 26]]

  [[ 1  1  1 ...,  0  4  0]]]]
"""

argmax1 = argmax1[0,:,0,:]
argmax2 = argmax2[0,:,0,:]
argmax3 = argmax3[0,:,0,:]

print ("argmax1"+str(argmax1))
print ("argmax2"+str(argmax2))
print ("argmax3"+str(argmax3))

"""
argmax1[[ 5  3  0 ...,  2  3  4]
 [10  9  1 ...,  5 12  0]
 [11 13  6 ..., 19  9  9]
 ..., 
 [ 9 13  4 ..., 14  6  5]
 [ 8 17  9 ...,  4 38 12]
 [ 2  0  0 ...,  4  0  1]]
argmax2[[ 2  1  2 ...,  2  1  1]
 [ 3  2  0 ...,  0  6 12]
 [15 11 13 ..., 12 11 11]
 ..., 
 [13  6  3 ..., 14  6  6]
 [ 3 36 25 ..., 18 10 39]
 [ 2  1  2 ...,  2  1  0]]
argmax3[[ 5  0  2 ...,  0  5  0]
 [ 0  0  4 ...,  5 13  3]
 [ 9  4  3 ..., 12  8  8]
 ..., 
 [12 12  0 ...,  1  7  6]
 [ 8 13 25 ..., 16 16 26]
 [ 1  1  1 ...,  0  4  0]]
"""
top_num = 15

argmax_sent_1 = np.zeros((top_num,100))
argmax_sent_2 = np.zeros((top_num,100))
argmax_sent_3 = np.zeros((top_num,100))

maximum1 = np.load("maximum_filter_sizes_1.npy")
maximum2 = np.load("maximum_filter_sizes_2.npy")
maximum3 = np.load("maximum_filter_sizes_3.npy")

maximum1 = maximum1[0,:,0,:]
maximum2 = maximum2[0,:,0,:]
maximum3 = maximum3[0,:,0,:]

for i in range(100):
	argmax_sent_1[:,i] = maximum1[:,i].argsort()[-top_num:][::-1]
	argmax_sent_2[:,i] = maximum2[:,i].argsort()[-top_num:][::-1]
	argmax_sent_3[:,i] = maximum3[:,i].argsort()[-top_num:][::-1]

argmax_sent_1 = argmax_sent_1.astype(int)
argmax_sent_2 = argmax_sent_2.astype(int)
argmax_sent_3 = argmax_sent_3.astype(int)

labels = ['pos','neg']

positions_1 = np.zeros(shape = (top_num, embeddings.shape[1]))
distance_1 = np.zeros(shape = (top_num, embeddings.shape[1]))

positions_2	= np.zeros(shape = (top_num, embeddings.shape[1]))
distance_2 = np.zeros(shape = (top_num, embeddings.shape[1]))

positions_3 = np.zeros(shape = (top_num, embeddings.shape[1]))
distance_3 = np.zeros(shape = (top_num, embeddings.shape[1]))

prediction_labels = np.load("gold_labels.npy")

for i in range(argmax1.shape[1]):
	f = open("/proj/zhou/sentiment/mycode/feature_maps_"+ flag +"/fm_filter_sizes_1_"+str(i+1), "w")
	argmaxs = argmax1[:,i]
	
	embeddings_plot = np.empty(shape = (0,embeddings.shape[1]))
	embeddings_label = []
	ngrams = []


	distances = 1234 * np.ones(top_num)
	labels_stats = [" "]*top_num

	for k in range(argmax_sent_1.shape[0]):
		#argmax_sent_1.shape[0] = 15
		#for top 15 sentences
		#j is sentence number 
		j = argmax_sent_1[k,i]

		f.write(labels[np.argmax(test_label[j,:])]) 
		f.write("\t\t")
		f.write(labels[prediction_labels[j]])
		f.write("\t\t")
		labels_stats[k] = labels[prediction_labels[j]]

		argmax = int(argmaxs[j])

		sent = np.asarray(test_dataset[j,:])
		"""
		modal_position = np.where(sent==modal_index)[0]

		if (len(modal_position) == 1):
			modal_position = int(modal_position)
			distances[k] = argmax - modal_position 
		"""	
		ngram = test_dataset[j,argmax:argmax+filter_sizes_1]

		pad = vocabulary["<PAD/>"]

		for word in ngram:
			if(word != pad):
				f.write(str(vocabulary_inv[word]) + " ")
		f.write("\t\t")

		embedding_ngram = np.zeros((1,embeddings.shape[1]))
		ngram_str = " "
		for word in ngram:
			try:
				embedding_word = embeddings[word,:].reshape((1,embeddings.shape[1]))
				embedding_ngram = embedding_ngram + embedding_word
				ngram_str = ngram_str + str(vocabulary_inv[word]) + " "
			except KeyError:
				#pass
				print (word)

		if (str(labels[np.argmax(test_label[j,:])]) != str(labels[prediction_labels[j]])):		
			ngram_str = ngram_str + "(" + str(labels[np.argmax(test_label[j,:])]) + ")"

		embeddings_plot = np.concatenate([embeddings_plot, embedding_ngram], 0)
		embeddings_label.append(prediction_labels[j])
		ngrams.append(ngram_str)

		for word in test_dataset[j,:]:
			if(word != pad):
				f.write(str(vocabulary_inv[word]) + " ")
		f.write("\n\n")
		
		#tsne = TSNE(init='pca')
		#two_d_embeddings = tsne.fit_transform(embeddings_plot)

		#file_path = "/home/mitarb/marasovic/CNN/feature_maps_"+ flag + "_" + modal_verb+"/fm_filter_sizes_1_"+str(i+1)+".png"
		#plot(two_d_embeddings , embeddings_label, ngrams, file_path)
		"""
		distances_invalid = distances[distances == 1234]
		f.write("number_of_sent_among_top_15_with_two_mvs\t" + str(distances_invalid.size) + "\n\n\n")
		distances_all_invalid.append(distances_invalid.size)

		distances_valid = distances[distances != 1234]
		f.write("avg_distance_from_mv_among_top_valid_15\t" + str(np.average(np.absolute(distances_valid))) + "\n\n\n")
		if (np.isnan(np.average(np.absolute(distances_valid))) == False):
			distances_all_valid.append(np.average(np.absolute(distances_valid)))

		distances_right = distances_valid[distances_valid > 0]
		f.write("number of ngrams on the right from mv\t" + str(distances_right.size) + "\n\n\n")
		distances_all_num_right.append(distances_right.size)
		f.write("avg_distance_from_mv_among_top_valid_15_right\t" + str(np.average(np.absolute(distances_right))) + "\n\n\n")
		if (np.isnan(np.average(np.absolute(distances_right))) == False):
			distances_all_right.append(np.average(np.absolute(distances_right)))

		distances_left = distances_valid[distances_valid < 0]
		for k in range(distances_left.size):
			distances_left[k] = distances_left[k] + float(filter_sizes_1 - 1)
		f.write("number of ngrams on the left from mv\t" + str(distances_left.size) + "\n\n\n")
		distances_all_num_left.append(distances_left.size)
		f.write("avg_distance_from_mv_among_top_valid_15_left\t" + str(np.average(np.absolute(distances_left))) + "\n\n\n")
		if (np.isnan(np.average(np.absolute(distances_left))) == False):
			distances_all_left.append(np.average(np.absolute(distances_left)))

		distances_exactly = distances_valid[distances_valid == 0]
		f.write("number of ngrams starting exactly on mv\t" + str(distances_exactly.size) + "\n\n\n")
		distances_all_num_exactly.append(distances_exactly.size)
		f.write("avg_distance_from_mv_among_top_valid_15_exact\t" + str(np.average(np.absolute(distances_exactly))) + "\n\n\n")
		if (np.isnan(np.average(np.absolute(distances_exactly))) == False):
			distances_all_exactly.append(distances_exactly)

		labels_file = ['pos','neg']
		labels_stats = np.asarray(labels_stats)

		f.write("-----"+"\n\n\n")

		for i,label in enumerate(labels_file):
			indices = np.where(labels_stats == label)
			distances_label = distances[indices]
			distances_label = distances_label[distances_label != 1234]
			f.write("avg_distance_from_mv_among_top_valid_15\t" + label + "\t" + str(np.average(np.absolute(distances_label))) + "\n\n\n")
			if (np.isnan(np.average(np.absolute(distances_label))) == False):
				distances_all_valid_label[i].append(np.average(np.absolute(distances_label)))

			distances_right = distances_label[distances_label < 0]
			f.write("number of ngrams on the right from mv\t" + label + "\t" + str(distances_right.size) + "\n\n\n")
			f.write("avg_distance_from_mv_among_top_valid_15_right\t" + label + "\t" + str(np.average(np.absolute(distances_right))) + "\n\n\n")
			distances_all_num_right_label[i].append(distances_right.size)
			if (np.isnan(np.average(np.absolute(distances_right))) == False):
				distances_all_right_label[i].append(np.average(np.absolute(distances_right)))

			distances_left = distances_label[distances_label > 0]

			for k in range(distances_left.size):
				distances_left[k] = distances_left[k] + float(filter_sizes_1 -1)
			f.write("number of ngrams on the left from mv\t" + label + "\t" + str(distances_left.size) + "\n\n\n")
			f.write("avg_distance_from_mv_among_top_valid_15_left\t" + label + "\t" + str(np.average(np.absolute(distances_right))) + "\n\n\n")
			distances_all_num_left_label[i].append(distances_left.size)
			if (np.isnan(np.average(np.absolute(distances_left))) == False):
				distances_all_left_label[i].append(np.average(np.absolute(distances_left)))


			distances_exactly = distances_label[distances_label == 0]
			f.write("number of ngrams starting exactly on mv\t" + label + "\t" + str(distances_exactly.size) + "\n\n\n")
			f.write("avg_distance_from_mv_among_top_valid_15_exact\t" + label + "\t" + str(np.average(np.absolute(distances_exactly))) + "\n\n\n")
			distances_all_num_exactly_label[i].append(distances_exactly.size)
			if (np.isnan(np.average(np.absolute(distances_exactly))) == False):
				distances_all_exactly_label[i].append(np.average(np.absolute(distances_exactly)))
		"""


		#np.save('/home/mitarb/marasovic/CNN/feature_maps_stats/' + modal_verb + '/dist_filter_sizes_1_' +str(i+1)+".npy", distances)
		#np.save('/home/mitarb/marasovic/CNN/feature_maps_stats/' + modal_verb + '/labels_filter_sizes_1_' +str(i+1)+".npy", distances)
		#f.close()
	
	

