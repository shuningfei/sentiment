"""
Reference code:
1.http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
2.https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/4_convolutions.ipynb
3.Ana Marasovic's code for MSC, of publication Ana Marasovic and Anette Frank (2016): Multilingual modal sense classification using a convolutional neural network. In Proceedings of the 1st Workshop on Representation Learning for NLP, Berlin, Germany
"""

import re
from collections import Counter
import itertools
from gensim import models
import numpy as np
import pickle

def clean_str(string):
    """
    Tokenization/string cleaning; original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data(dataset_file):
	dataset = list(open(dataset_file,"rb").readlines())
	dataset = [s.decode('latin-1').strip() for s in dataset]

	dataset_text = [clean_str(sent) for sent in dataset]
	dataset_text = [s.split(" ") for s in dataset_text]

	
	return dataset_text

# if real sentence length < sentence_length, add with "<PAD/>"
def pad_sentences(sentences, sentence_length, padding_word="<PAD/>"):
	"""
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
	padded_sentences = []
	for i in range(len(sentences)):
		sentence = sentences[i]
		num_padding = sentence_length - len(sentence)
		new_sentence = sentence + [padding_word] * num_padding
		padded_sentences.append(new_sentence)
	return padded_sentences

def build_vocab_and_embeddings(sentences, vector, vocab_embedding_pickle):
	print ("Building vocabulary...")
	# Build vocabulary
	word_counts = Counter(itertools.chain(*sentences))
	# Mapping from index to word
	vocabulary_inv = [x[0] for x in word_counts.most_common()]
	# Mapping from word to index
	vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

	if(vector == 'w2v'):
		print ("Loading w2v model...")
		model = models.Word2Vec.load_word2vec_format('../../CNN/GoogleNews-vectors-negative300.bin', binary = True)

		print ("Building embeddings...")
		vocab_size = len(vocabulary)
		embeddings = np.zeros((vocab_size, 300))
		# a matrix of zero => 300 * vocab_size
		
		for word in vocabulary:
			index = vocabulary[word]
			try:
				embeddings[index, :] = model[word].reshape((1,300))
			except KeyError:
				embeddings[index, :] = np.random.uniform(-0.23, 0.23, [1,300])
		# -0.23,0.23 means number between -0.23 and 0.23
		# every word in vocabulary is 300 * 1, if w2v model contans word, uses model, if not, uses random.

		print ("Write data in a pickle...")
		pickle_file = 'w2v_'+vocab_embedding_pickle+'.pickle'
		try:
			fp = open(pickle_file, 'wb')
			save = {
				'vocabulary': vocabulary,
				'embeddings': embeddings
			}
				
			pickle.dump(save, fp, pickle.HIGHEST_PROTOCOL)
			fp.close()
		except Exception as e:
			print ('Unable to save data to', pickle_file, ':', e)
			raise
		
	if (vector == 'random'):
		vocab_size = len(vocabulary)
		embeddings = np.random.uniform(-1.0, 1.0, [vocab_size, 300])
			
		print ("Write data in a pickle...")
		pickle_file = 'random_'+vocab_embedding_pickle+'.pickle'
		try:
			fp = open(pickle_file, 'wb')
			save = {
				'vocabulary': vocabulary,
				'embeddings': embeddings
			}
			pickle.dump(save, fp, pickle.HIGHEST_PROTOCOL)
			fp.close()
		except Exception as e:
			print ('Unable to save data to', pickle_file, ':', e)
			raise

if __name__ == "__main__":

	pos_dataset_file = "classifier1/positive-a1.txt"
	neg_dataset_file = "classifier1/negative-a1.txt"


	sentences_pos = load_data(pos_dataset_file)
	sentence_length_pos = max(len(x) for x in sentences_pos)
	
	sentences_neg = load_data(neg_dataset_file)
	sentence_length_neg = max(len(x) for x in sentences_neg)
	
	sentence_length = max(sentence_length_pos, sentence_length_neg)
	# print (sentence_length)
	# sentence_length: 185
	
	sentences_padded_pos = pad_sentences(sentences_pos, sentence_length)
	sentences_padded_neg = pad_sentences(sentences_neg, sentence_length)
	
	sentences_all = sentences_padded_pos + sentences_padded_neg
	
	vectors = ['w2v', 'random']
	
	for vector in vectors:
		build_vocab_and_embeddings(sentences_all, vector,'class1-a1')