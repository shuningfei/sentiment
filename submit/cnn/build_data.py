"""
Reference code:
1.http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
2.https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/4_convolutions.ipynb
3.Ana Marasovic's code for MSC, of publication Ana Marasovic and Anette Frank (2016): Multilingual modal sense classification using a convolutional neural network. In Proceedings of the 1st Workshop on Representation Learning for NLP, Berlin, Germany
"""

import pickle
from build_vocab_embed import clean_str,pad_sentences
import numpy as np

def load_data_and_labels(positiveData,negativeData):
	"""
	Loads MR polarity data from files, splits the data into words and generates labels.
	Returns split sentences and labels.
	"""
	# Load data from files
	positive_examples = list(open(positiveData, "r").readlines())
	positive_examples = [s.strip() for s in positive_examples]
	negative_examples = list(open(negativeData, "r").readlines())
	negative_examples = [s.strip() for s in negative_examples]
	# Split by words
	x = positive_examples + negative_examples
	x = [clean_str(sent) for sent in x]
	x = [s.split(" ") for s in x]
	# Generate labels
	positive_labels = [[0, 1] for _ in positive_examples]
	negative_labels = [[1, 0] for _ in negative_examples]
	y = np.concatenate([positive_labels, negative_labels], 0)
	return [x, y]

def build_input_data(sentences, labels, vocabulary):
	"""
	Maps sentencs and labels to vectors based on a vocabulary.
	"""
	x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
	y = np.array(labels)
	return [x, y]
	
def shuffle_split_data(data,label):
	
	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(len(label)))
	
	data_shuffled = data[shuffle_indices]
	label_shuffled = label[shuffle_indices]

	testLength = int(len(data_shuffled)/10)

	train_data,test_data = data_shuffled[:-testLength],data_shuffled[-testLength:]
	train_label,test_label = label_shuffled[:-testLength], label_shuffled[-testLength:]

	return [train_data,test_data,train_label,test_label]

def shuffle_data(data,label):

	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(len(label)))

	data_shuffled = data[shuffle_indices]
	label_shuffled = label[shuffle_indices]

	return [data_shuffled,label_shuffled]


def saveData(sentence_length,vocab_embedding_pickle,positiveData,negativeData):

	filename = 'data_shuffled'
	f = open(filename,'a')

	sentences, labels = load_data_and_labels(positiveData,negativeData)

	pickle_file = "w2v_"+vocab_embedding_pickle+'.pickle'

	print("Loading vocabulary...")

	try:
		with open(pickle_file,'rb') as fp:
			save = pickle.load(fp)
			vocabulary = save['vocabulary']
			del save
	except EOFError:
		print ('Unable to do something')

	pad_sent = pad_sentences(sentences, sentence_length)
	dataset, label = build_input_data(pad_sent, labels, vocabulary)
	train_dataset,test_dataset,train_label,test_label = shuffle_split_data(dataset,label)

	f.write("Train data shape: " + str(np.shape(train_dataset)) + "\n\n")
	f.write("Test data shape: " + str(np.shape(test_dataset)) + "\n\n")

	vectors_list = ['w2v', 'random']

	for vectors in vectors_list:
		pickle_file = vectors + "_" + vocab_embedding_pickle + ".pickle"
		print("Loading vocabulary and embeddings...")

		try:
			with open(pickle_file, 'rb') as fp:
				save = pickle.load(fp)
				vocabulary = save['vocabulary']
				embeddings = save['embeddings']
				del save
		except EOFError:
			print ('Unable to do something')

		print ("Write data in a pickle...")
		pickle_file = vectors + "_" + vocab_embedding_pickle + '_data.pickle'

		try:
			fp = open(pickle_file, 'wb')
			save = {
				'train_dataset': train_dataset,
				'train_label': train_label,
				'test_dataset': test_dataset,
				'test_label': test_label,
				'vocabulary': vocabulary,
				'embeddings': embeddings
			}
			pickle.dump(save, fp, pickle.HIGHEST_PROTOCOL)
			fp.close()
		except Exception as e:
			print ('Unable to save data to', pickle_file, ':', e)
			raise


if __name__=='__main__':

	saveData(185,'class1-a1',"classifier1/positive-a1.txt","classifier1/negative-a1.txt")
