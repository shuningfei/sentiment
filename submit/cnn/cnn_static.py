"""
reference code:
1. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/4_convolutions.ipynb
2. http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

CNN-static: Embedding matrix keep static, filter matrix and bias term are learned. tf.Variable means learned parameters.

"""

import numpy as np
import tensorflow as tf

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


class TextCNN(object):
	def __init__(self,train_dataset, train_labels, valid_dataset, valid_labels, embeddings, vocabulary, l2_reg_lambda, num_steps, batch_size, num_filters, filter_sizes_1, filter_sizes_2, filter_sizes_3, dropout_keep_prob):

		# parameters
		sequence_length = train_dataset.shape[1]
		# maximum sentence length, 185 in this case
		num_classes = 2
		# neg and pos
		filter_sizes = [filter_sizes_1, filter_sizes_2, filter_sizes_3]
		# [3,4,5]
		num_filters_total = num_filters * len(filter_sizes)
		# 3*100 = 300
		embedding_size = embeddings.shape[1]
		# 300 dimensions
		embeddings_number = embeddings.shape[0]
		# 3931, embeddings number is same as vocabulary size
		graph = tf.Graph()
		with graph.as_default():
		# First you describe the computation that you want to see performed: what the inputs, the variables, and the operations look like. These get created as nodes over a computation graph

			tf.set_random_seed(10)
			# avoid different runs with the same settings provide different results

			#variables and constants
			input_x = tf.placeholder(tf.int32, shape = [batch_size, sequence_length])
			input_y = tf.placeholder(tf.int32, shape = [batch_size, num_classes])

			tf_valid_dataset = tf.constant(valid_dataset)

			reg_coef = tf.placeholder(tf.float32)
			l2_loss = tf.constant(0.0)

			weights_conv = [tf.Variable(tf.truncated_normal([filter_size, embedding_size, 1, num_filters], stddev = 0.1, seed = filter_size + i*num_filters)) for i, filter_size in enumerate(filter_sizes)]
			# this is our filter matrix (weight matrix)
			# filter_shape = [filter_size, embedding_size, 1, num_filters] = [3,300,1,100] first two dimensions are the size of the window, third is the amount of channels, which is 1 in our case, last one defines how many features we want to use.
    		# truncated_normal: random values with a normal distribution but eliminating those values whose magnitude is more than 2 times the standard deviation.
    		# this is a list

			biases_conv = [tf.Variable(tf.constant(0.01, shape=[num_filters])) for filter_size in filter_sizes]
			# we will also need to define a bias for every of 100 weight matrices.
    		# this is also a list

			weight_output = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev = 0.1))
			bias_output = tf.Variable(tf.constant(0.01, shape=[num_classes]))
			# weights_out: [3*100,2]
    		# biases_out: [2]
    		# will be fed to a final softnax layer of 2 classes

			embeddings_const = tf.placeholder(tf.float32, shape = [embeddings_number, embedding_size])
			# embedding matrix that we learn during training.
    		# will be fed to embeddings as initialization (random or w2v)

			embedded_chars = tf.nn.embedding_lookup(embeddings_const, input_x)
			embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
			# embedding_lookup creates the actual embedding operation

			embedded_chars_valid = tf.nn.embedding_lookup(embeddings_const, tf_valid_dataset)
			embedded_chars_expanded_valid = tf.expand_dims(embedded_chars_valid, -1)

			# model: convolution and max-pooling, for more see http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/:
			def model(data, dropout_prob):
				pooled_outputs = []
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

			# Training computation.
			scores = model(embedded_chars_expanded, dropout_keep_prob)
			losses = tf.nn.softmax_cross_entropy_with_logits(scores, tf.cast(input_y, tf.float32))

			for i in range(len(weights_conv)):
				l2_loss += tf.nn.l2_loss(weights_conv[i])
			l2_loss += tf.nn.l2_loss(weight_output)

			loss = tf.reduce_mean(losses) + reg_coef * l2_loss

			# Optimizer.
			optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

			# Predictions for the training, validation, and test data.
			train_prediction = tf.nn.softmax(scores)
			valid_prediction = tf.nn.softmax(model(embedded_chars_expanded_valid, 1.0))

		# Then you can run the operations on this graph as many times as you want by calling session.run(), providing it outputs to fetch from the graph that get returned
		with tf.Session(graph=graph) as session:
			tf.initialize_all_variables().run()
			print ("Initialized")

			for step in range(num_steps):
				offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
				batch_data = train_dataset[offset:(offset + batch_size)]
				batch_labels = train_labels[offset:(offset + batch_size)]
				feed_dict = {input_x : batch_data, input_y : batch_labels, reg_coef: l2_reg_lambda, embeddings_const: embeddings}
				_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict)

				if not step % 100:
					print ("Minibatch loss at step", step, ":", l)
					print ("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
					print("\n")

			# learn parameters, then it predicts. the output of softmax is probability, e.g [ 0.04484121  0.95515883], sum is 1.
			self.valid_predictions = session.run([valid_prediction], feed_dict = {embeddings_const: embeddings})
			# self.valid_predictions.shape for test data: (1,220,2)

			self.valid_predictions = np.asarray(self.valid_predictions).reshape(valid_labels.shape)
			# valid_labels.shape: (220,2)

			self.valid_accuracy = accuracy(self.valid_predictions, np.asarray(valid_labels))
			self.embeddings_final = embeddings

