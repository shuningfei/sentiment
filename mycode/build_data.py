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

def shuffle_add_train_data(data,label,train_data,test_data,train_label,test_label):

	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(len(label)))

	data_shuffled = data[shuffle_indices]
	label_shuffled = label[shuffle_indices]

	train_data = np.concatenate((train_data,data_shuffled))
	train_label = np.concatenate((train_label,label_shuffled))

	return [train_data,test_data,train_label,test_label]

def shuffle_data(data,label):

	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(len(label)))

	data_shuffled = data[shuffle_indices]
	label_shuffled = label[shuffle_indices]

	return [data_shuffled,label_shuffled]



"""
# from Ana's MSC -> can:ep,de,dy  
def load_data_and_labels(sentences_file):
	lines = open(sentences_file,"r", encoding = 'latin-1').readlines()
	
	sentences_list = []
	labels_list = []

	for line in lines:
		line_split = line.split("\t")
		sentences_list.append(line_split[0])
		features = line_split[3].split(",")
		labels_list.append(features[len(features)-1].split('\n')[0])

	sentences = [clean_str(sent) for sent in sentences_list]
	sentences = [s.split(" ") for s in sentences]

	num_of_classes = 3 
	labels_set = {'ep': 0, 'de': 1, 'dy': 2}

	labels = []

	for label in labels_list:
		temp = [0]*num_of_classes
		index = labels_set[label]
		temp[index] = 1
		labels.append(temp)
	# labels: 3 dimension vector, if 'ep', then [1,0,0]; if 'de', then [0,1,0]

	labels = np.asarray(labels)
	return [sentences, labels]
"""

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

def saveDataV2(sentence_length,vocab_embedding_pickle,positiveData,negativeData,positiveData2,negativeData2):

	filename = 'data_shuffled'
	f = open(filename,'a')

	sentences, labels = load_data_and_labels(positiveData,negativeData)
	sentencesV2, labelsV2 = load_data_and_labels(positiveData2,negativeData2)

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

	pad_sentV2 = pad_sentences(sentencesV2,sentence_length)
	datasetV2, labelV2 = build_input_data(pad_sentV2,labelsV2,vocabulary)
	train_dataset,test_dataset,train_label,test_label = shuffle_add_train_data(datasetV2,labelV2,train_dataset,test_dataset,train_label,test_label)

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

def saveDataV3(sentence_length,vocab_embedding_pickle,positiveData_train,negativeData_train,positiveData_test,negativeData_test):

	filename = 'data_shuffled'
	f = open(filename,'a')

	sentences_train, labels_train = load_data_and_labels(positiveData_train,negativeData_train)
	sentences_test, labels_test = load_data_and_labels(positiveData_test,negativeData_test)

	pickle_file = "w2v_"+vocab_embedding_pickle+'.pickle'

	print("Loading vocabulary...")

	try:
		with open(pickle_file,'rb') as fp:
			save = pickle.load(fp)
			vocabulary = save['vocabulary']
			del save
	except EOFError:
		print ('Unable to do something')

	pad_sent_train = pad_sentences(sentences_train, sentence_length)
	dataset, label = build_input_data(pad_sent_train, labels_train, vocabulary)
	train_dataset,train_label = shuffle_data(dataset,label)

	pad_sent_test = pad_sentences(sentences_test,sentence_length)
	dataset2, label2 = build_input_data(pad_sent_test, labels_test, vocabulary)
	test_dataset,test_label = shuffle_data(dataset2,label2)

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

	saveData(185,'class3subj-a1',"classifier3/subj_sents-a1.txt","classifier3/not_subj_sents-a1.txt")

	#saveDataV2(185,'class1b-a1',"classifier1/relCorReplaced-pos-a1.txt","classifier1/relCorReplaced-neg-a1.txt","paraphrases/subjective-pos-a1.txt","paraphrases/subjective-neg-a1.txt")

	#saveDataV3(185,'class1c-a1',"classifier1c/sentPara-train-pos-a1.txt","classifier1c/sentPara-train-neg-a1.txt","classifier1c/relCorRep-pos-test-a1.txt","classifier1c/relCorRep-neg-test-a1.txt")

