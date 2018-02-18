import pickle
from build_vocab_embed import clean_str,pad_sentences
import numpy as np


def load_data_and_labels():

	products = ['coffeemachine','cutlery','microwave','toaster','trashcan','vacuum','washer']

	x = []

	for prod in products:

		prod_dataset_file = "classifier4/"+prod+"-a1.txt"
		prod_examples = list(open(prod_dataset_file,"r").readlines())
		prod_examples = [s.strip() for s in prod_examples]
		x += prod_examples

	x = [clean_str(sent) for sent in x]
	x = [s.split(" ") for s in x]

	labels = []

	num_of_classes = 7
	labels_set = {'coffeemachine': 0, 'cutlery': 1, 'microwave': 2, 'toaster': 3, 'trashcan': 4, 'vacuum': 5, 'washer': 6}

	for prod in products:
		temp = [0]*num_of_classes
		index = labels_set[prod]
		temp[index] = 1
		prod_labels = [temp for _ in prod_examples]
		labels.extend(prod_labels)

	y = np.asarray(labels)

	return [x,y]

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

	testLength = int(len(data_shuffled)/5)

	train_data,test_data = data_shuffled[:-testLength],data_shuffled[-testLength:]
	train_label,test_label = label_shuffled[:-testLength], label_shuffled[-testLength:]

	return [train_data,test_data,train_label,test_label]


def saveData(sentence_length,vocab_embedding_pickle):

	filename = 'data_shuffled'
	f = open(filename,'a')

	sentences, labels = load_data_and_labels()

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

	saveData(185,'class4-a1')

	#saveData(185,'class1-a1',"classifier1/relCorReplaced-pos-a1.txt","classifier1/relCorReplaced-neg-a1.txt")

	#saveDataV2(185,'class1b-a1',"classifier1/relCorReplaced-pos-a1.txt","classifier1/relCorReplaced-neg-a1.txt","paraphrases/subjective-pos-a1.txt","paraphrases/subjective-neg-a1.txt")

	#saveDataV3(185,'class1c-a1',"classifier1c/sentPara-train-pos-a1.txt","classifier1c/sentPara-train-neg-a1.txt","classifier1c/relCorRep-pos-test-a1.txt","classifier1c/relCorRep-neg-test-a1.txt")

