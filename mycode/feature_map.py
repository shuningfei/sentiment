import sys
import tensorflow as tf
import pickle
import numpy as np

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def model(data,dropout_prob,weights_conv_l,biases_conv_l,weight_output,bias_output):
	pooled_outputs = []
	for i, filter_size in enumerate(filter_sizes):
		conv = tf.nn.conv2d(data,weights_conv_l[i],strides=[1, 1, 1, 1], padding="VALID")
		h = tf.nn.relu(tf.nn.bias_add(conv, biases_conv_l[i]))
		pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
		pooled_outputs.append(pooled)
		
	h_pool = tf.concat(3,pooled_outputs)
	h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
	h_drop = tf.nn.dropout(h_pool_flat, dropout_prob)
	scores = tf.nn.xw_plus_b(h_drop, weight_output, bias_output)
	return scores

def model_argmax(data,dropout_prob,weights_conv_l,biases_conv_l):
	argmaxs = []
	maximums = []
	pooled_outputs = []
	for i, filter_size in enumerate(filter_sizes):
		conv = tf.nn.conv2d(data, weights_conv_l[i], strides=[1, 1, 1, 1], padding="VALID")
		h = tf.nn.relu(tf.nn.bias_add(conv, biases_conv_l[i]))
		maximum = tf.reduce_max(h,tf.to_int32(1))
		maximums.append(maximum)
		argmax = tf.argmax(h, tf.to_int32(1))
		argmaxs.append(argmax)
	return (argmaxs, maximums)
	
		
labels = ['pos','neg']

flag = sys.argv[1]
# flag: train, test

vectors = "w2v"
tuning = "static"
lexical = "lex"

pickle_file = vectors + '_data.pickle'

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

if (flag == "train"):
	test_dataset = train_dataset
	test_label = train_label
	
	train_dataset = train_dataset
	train_labels = train_label
	valid_dataset = test_dataset
	valid_labels = test_label
	embeddings = embeddings
	vocabulary = vocabulary
	l2_reg_lambda = l2_reg_lambda
	batch_size = batch_size
	num_filters = num_filters
	filter_sizes_1 = filter_sizes_1
	filter_sizes_2 = filter_sizes_2
	filter_sizes_3 = filter_sizes_3
	dropout_keep_prob = dropout_keep_prob
	lexical = "lex"
	shuffling = "n"
	
	vocab_size = len(vocabulary)
	sequence_length = train_dataset.shape[1]
	train_size = train_dataset.shape[0]
	num_classes = 2
	
	filter_sizes = [filter_sizes_1,filter_sizes_2,filter_sizes_3]
	num_filters_total = num_filters * len(filter_sizes)
	
	embedding_size = embeddings.shape[1]
	embeddings_number = embeddings.shape[0]
	
	graph = tf.Graph()
	with graph.as_default():
		tf.set_random_seed(10)
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
		
		embedded_chars = tf.nn.embedding_lookup(embeddings_const, input_x)
		embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
		
		embedded_chars_valid = tf.nn.embedding_lookup(embeddings_const, tf_valid_dataset)
		embedded_chars_expanded_valid = tf.expand_dims(embedded_chars_valid, -1)
	
		embeddings_tuned_argmax = tf.placeholder(tf.float32, shape = [None, embedding_size])
		embedded_chars_argmax = tf.nn.embedding_lookup(embeddings_tuned_argmax, tf_argmax_dataset)
		embedded_chars_expanded_argmax = tf.expand_dims(embedded_chars_argmax, -1)
		
		scores = model(embedded_chars_expanded, dropout_keep_prob,weights_conv,biases_conv,weight_output,bias_output)
		train_prediction = tf.nn.softmax(scores)	
		
		losses = tf.nn.softmax_cross_entropy_with_logits(scores, tf.cast(input_y, tf.float32))
		
		for i in range(len(weights_conv)):
			l2_loss += tf.nn.l2_loss(weights_conv[i])
		l2_loss += tf.nn.l2_loss(weight_output)
		
		loss = tf.reduce_mean(losses) + reg_coef * l2_loss
		
		global_step = tf.Variable(0, trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
		
		argmaxs, maximums = model_argmax(embedded_chars_expanded_argmax, 1.0,weights_conv,biases_conv)
		maximum1 = maximums[0]
		maximum2 = maximums[1]
		maximum3 = maximums[2]
		argmax1 = argmaxs[0]
		argmax2 = argmaxs[1]
		argmax3 = argmaxs[2]
			
		valid_prediction = tf.nn.softmax(model(embedded_chars_expanded_valid, 1.0,weights_conv,biases_conv,weight_output,bias_output))
		
	
	with tf.Session(graph=graph) as session:
		#session.run(tf.initialize_all_variables(), feed_dict={embeddings_const: embeddings})
		session.run(tf.initialize_all_variables())
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
		
		maximum1 = session.run([maximum1], feed_dict = {embeddings_tuned_argmax: embeddings})
		maximum1 = np.asarray(maximum1)
		maximum2 = session.run([maximum2], feed_dict = {embeddings_tuned_argmax: embeddings})
		maximum2 = np.asarray(maximum2)
		maximum3 = session.run([maximum3], feed_dict = {embeddings_tuned_argmax: embeddings})
		maximum3 = np.asarray(maximum3)

		argmax1 = session.run([argmax1], feed_dict = {embeddings_tuned_argmax: embeddings})
		argmax1 = np.asarray(argmax1)
		argmax2 = session.run([argmax2], feed_dict = {embeddings_tuned_argmax: embeddings})
		argmax2 = np.asarray(argmax2)
		argmax3 = session.run([argmax3], feed_dict = {embeddings_tuned_argmax: embeddings})
		argmax3 = np.asarray(argmax3)

		np.save("argmax_filter_sizes_1.npy", argmax1)
		np.save("argmax_filter_sizes_2.npy", argmax2)
		np.save("argmax_filter_sizes_3.npy", argmax3)

		np.save("maximum_filter_sizes_1.npy", maximum1)
		np.save("maximum_filter_sizes_2.npy", maximum2)
		np.save("maximum_filter_sizes_3.npy", maximum3)
				
		valid_predictions = session.run([valid_prediction], feed_dict = {embeddings_const: embeddings})
		valid_predictions = np.asarray(valid_predictions).reshape(valid_labels.shape)
		
		predictions_label = np.argmax(valid_predictions, 1)		
		labels = ['pos','neg']
		
		prediction_labels_char = [labels[i] for i in predictions_label]
		prediction_labels_char = np.asarray(prediction_labels_char)
		
		np.save("gold_labels.npy", predictions_label)
		
		valid_accuracy = accuracy(valid_predictions, np.asarray(valid_labels))
		
		print (valid_accuracy)
		
		embeddings_final = embeddings

			
			
			
			
			
			