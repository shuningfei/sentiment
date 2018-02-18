"""
reference code:
1. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/4_convolutions.ipynb
2. http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
3.Ana Marasovic's code for MSC, of publication Ana Marasovic and Anette Frank (2016): Multilingual modal sense classification using a convolutional neural network. In Proceedings of the 1st Works
"""

import sys
import numpy as np
import tensorflow as tf
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn import metrics
    
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


class TextCNN(object):
	def __init__(self,train_dataset, train_labels, valid_dataset, valid_labels, embeddings, vocabulary, l2_reg_lambda, num_steps, batch_size, num_filters, filter_sizes_1, filter_sizes_2, filter_sizes_3, dropout_keep_prob, lexical, shuffling):
		# parameters
		vocab_size = len(vocabulary)
		sequence_length = train_dataset.shape[1]
		train_size = train_dataset.shape[0]
		num_classes = 2

		filter_sizes = [filter_sizes_1, filter_sizes_2, filter_sizes_3]
		num_filters_total = num_filters * len(filter_sizes)

		embedding_size = embeddings.shape[1]
		embeddings_number = embeddings.shape[0]

		graph = tf.Graph()
		with graph.as_default():
			tf.set_random_seed(10)
			#variables and constants
			input_x = tf.placeholder(tf.int32, shape = [batch_size, sequence_length])
			input_y = tf.placeholder(tf.int32, shape = [batch_size, num_classes])

			tf_valid_dataset = tf.constant(valid_dataset)
			tf_argmax_dataset = tf.constant(valid_dataset)

			reg_coef = tf.placeholder(tf.float32)

			l2_loss = tf.constant(0.0)

			weights_conv = [tf.Variable(tf.truncated_normal([filter_size, embedding_size, 1, num_filters], stddev = tf.sqrt(2.0 / (filter_size*embedding_size)), seed = filter_size + i*num_filters)) for i, filter_size in enumerate(filter_sizes)]
			biases_conv = [tf.Variable(tf.constant(0.01, shape=[num_filters])) for filter_size in filter_sizes]

			weight_output = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev = tf.sqrt(2.0 / (num_filters_total+num_classes)), seed = 0))
			bias_output = tf.Variable(tf.constant(0.01, shape=[num_classes]))

			embeddings_const = tf.placeholder(tf.float32, shape = [embeddings_number, embedding_size])
			embeddings_tuned = tf.Variable(embeddings_const)

			embedded_chars = tf.nn.embedding_lookup(embeddings_tuned, input_x)
			embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

			#embeddings_tuned_valid = tf.placeholder(tf.float32, shape = [None, embedding_size])
			embedded_chars_valid = tf.nn.embedding_lookup(embeddings_tuned, tf_valid_dataset)
			embedded_chars_expanded_valid = tf.expand_dims(embedded_chars_valid, -1)

			#embeddings_tuned_argmax = tf.placeholder(tf.float32, shape = [None, embedding_size])
			embedded_chars_argmax = tf.nn.embedding_lookup(embeddings_tuned, tf_argmax_dataset)
			embedded_chars_expanded_argmax = tf.expand_dims(embedded_chars_argmax, -1)

			def model(data, dropout_prob):
				pooled_outputs = []
				#lookup table
				for i, filter_size in enumerate(filter_sizes):
					#convolution layer with different filter size
					conv = tf.nn.conv2d(data, weights_conv[i], strides=[1, 1, 1, 1], padding="VALID")
					#non-linearity  
					h = tf.nn.relu(tf.nn.bias_add(conv, biases_conv[i]))
					pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
					pooled_outputs.append(pooled)

				h_pool = tf.concat(3, pooled_outputs)
				h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
				h_drop = tf.nn.dropout(h_pool_flat, dropout_prob)
				scores = tf.nn.xw_plus_b(h_drop, weight_output, bias_output)
				return scores

			def model_argmax(data, dropout_prob):
				argmaxs = []
				maximums = []
				pooled_outputs = []
				for i, filter_size in enumerate(filter_sizes):
					#sizes.append(filter_size)
					#convolution layer with different filter size
					conv = tf.nn.conv2d(data, weights_conv[i], strides=[1, 1, 1, 1], padding="VALID")
					#non-linearity  
					h = tf.nn.relu(tf.nn.bias_add(conv, biases_conv[i]))
					#pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
					#pooled_outputs.append(pooled)
					maximum = tf.reduce_max(h,tf.to_int32(1))
					maximums.append(maximum)
					argmax = tf.argmax(h, tf.to_int32(1))
					argmaxs.append(argmax)
				return (argmaxs, maximums)

			scores = model(embedded_chars_expanded, dropout_keep_prob)	
			train_prediction = tf.nn.softmax(scores)	

			losses = tf.nn.softmax_cross_entropy_with_logits(scores, tf.cast(input_y, tf.float32))

			for i in range(len(weights_conv)):
				l2_loss += tf.nn.l2_loss(weights_conv[i])
			l2_loss += tf.nn.l2_loss(weight_output)

			loss = tf.reduce_mean(losses) + reg_coef * l2_loss

			#global_step = tf.Variable(0)
			#learning_rate = tf.train.exponential_decay(1e-4, global_step * batch_size, tf.size(input_x), 0.95, staircase=True)
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

			global_step = tf.Variable(0, trainable=False)
			optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

			#global_step = tf.Variable(0, trainable=False)
			#optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)


			argmaxs, maximums = model_argmax(embedded_chars_expanded_argmax, 1.0)
			maximum1 = maximums[0]
			maximum2 = maximums[1]
			maximum3 = maximums[2]
			argmax1 = argmaxs[0]
			argmax2 = argmaxs[1]
			argmax3 = argmaxs[2]

			valid_prediction = tf.nn.softmax(model(embedded_chars_expanded_valid, 1.0))


		with tf.Session(graph=graph) as session:
			session.run(tf.initialize_all_variables(), feed_dict={embeddings_const: embeddings})
			print ("Initialized")
			
			for step in range(num_steps):
				offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
				batch_data = train_dataset[offset:(offset + batch_size)]
				batch_labels = train_labels[offset:(offset + batch_size)]
				feed_dict = {input_x : batch_data, input_y : batch_labels, reg_coef: l2_reg_lambda, embeddings_const: embeddings}
				_, l, predictions, embeddings = session.run([optimizer, loss, train_prediction, embeddings_tuned], feed_dict)
				# embedding changes
				
				
				if not step % 100:
					print ("Minibatch loss at step", step, ":", l)
					print ("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
					print("\n")

				#print (embeddings_after)
			
			maximum1 = session.run([maximum1], feed_dict = {embeddings_const: embeddings})
			maximum1 = np.asarray(maximum1)
			maximum2 = session.run([maximum2], feed_dict = {embeddings_const: embeddings})
			maximum2 = np.asarray(maximum2)
			maximum3 = session.run([maximum3], feed_dict = {embeddings_const: embeddings})
			maximum3 = np.asarray(maximum3)

			argmax1 = session.run([argmax1], feed_dict = {embeddings_const: embeddings})
			argmax1 = np.asarray(argmax1)
			argmax2 = session.run([argmax2], feed_dict = {embeddings_const: embeddings})
			argmax2 = np.asarray(argmax2)
			argmax3 = session.run([argmax3], feed_dict = {embeddings_const: embeddings})
			argmax3 = np.asarray(argmax3)

			np.save("argmax_filter_sizes_1_tuned.npy", argmax1)
			np.save("argmax_filter_sizes_2_tuned.npy", argmax2)
			np.save("argmax_filter_sizes_3_tuned.npy", argmax3)

			np.save("maximum_filter_sizes_1_tuned.npy", maximum1)
			np.save("maximum_filter_sizes_2_tuned.npy", maximum2)
			np.save("maximum_filter_sizes_3_tuned.npy", maximum3)

			self.valid_predictions = session.run([valid_prediction], feed_dict = {embeddings_const: embeddings})
			self.valid_predictions = np.asarray(self.valid_predictions).reshape(valid_labels.shape)	

			predictions_label = np.argmax(self.valid_predictions, 1)



			labels = ['neg','pos']

			self.prediction_labels_char = [labels[i] for i in predictions_label]
			self.prediction_labels_char = np.asarray(self.prediction_labels_char)

			np.save("gold_labels_tuned.npy", predictions_label)

			self.valid_accuracy = accuracy(self.valid_predictions, np.asarray(valid_labels))

			self.embeddings_final = embeddings
			
