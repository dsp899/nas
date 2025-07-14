import os
import csv
import time
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import utils


class model():
 
    def __init__(self,model_architecture,model_name):
        self.graph = tf.Graph()
        self.curdir = os.path.abspath(os.path.curdir)
        self.builtdir = '{}/{}'.format(self.curdir,'built')
        self.tensorflow_savedModels = '{}/{}'.format(self.builtdir,'tensorflow_savedModels')
        self.arch = model_architecture
        self.name = model_name

    def create_model_single(self, num_units, seq_length, num_features, num_classes):
        with self.graph.as_default():
            print("Create model_single")
            #self.inpnet = tf.placeholder(tf.float32, [None, seq_length, num_features], name='input')
            self.inpnet = tf.placeholder(tf.float32, [None, seq_length ,num_features[1], num_features[2], num_features[3]], name='input')

            self.labels = tf.placeholder(tf.float32, [None, num_classes], name='labels')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            avg_pool = tf.nn.avg_pool3d(self.inpnet, ksize=[1, 1 ,num_features[1], num_features[2], 1], strides=[1, 1, 1, 1, 1], padding='VALID')
            reduced_dim = tf.squeeze(avg_pool, axis=[2, 3])
            print(reduced_dim.shape)
            #reshape = tf.reshape(reduced_dim, shape=[-1,seq_length,reduced_dim.shape[2]*reduced_dim.shape[3]*reduced_dim.shape[4]])
            lstm_cell = tf.contrib.rnn.LSTMCell(num_units=num_units, dtype=tf.float32, name='lstm')
            outputs, states = tf.nn.dynamic_rnn(lstm_cell, reduced_dim, dtype=tf.float32, scope='lstm')
            #print(outputs.shape)
            dropout = tf.nn.dropout(outputs[:,-1,:], keep_prob=self.keep_prob, name='dropout')
            self.outnet = tf.layers.dense(dropout, num_classes,activation=tf.nn.relu, name='output')
    
    
    def create_model_single_bidirectional(self, num_units, seq_length, num_features, num_classes):
        with self.graph.as_default():
            print("Create model_single")
            #self.inpnet = tf.placeholder(tf.float32, [None, seq_length, num_features], name='input')
            self.inpnet = tf.placeholder(tf.float32, [None, seq_length ,num_features[1], num_features[2], num_features[3]], name='input')

            self.labels = tf.placeholder(tf.float32, [None, num_classes], name='labels')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            avg_pool = tf.nn.avg_pool3d(self.inpnet, ksize=[1, 1 ,num_features[1], num_features[2], 1], strides=[1, 1, 1, 1, 1], padding='VALID')
            reduced_dim = tf.squeeze(avg_pool, axis=[2, 3])
            print(reduced_dim.shape)
            #reshape = tf.reshape(reduced_dim, shape=[-1,seq_length,reduced_dim.shape[2]*reduced_dim.shape[3]*reduced_dim.shape[4]])
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(num_units=num_units, dtype=tf.float32, name='lstm_fw')
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(num_units=num_units, dtype=tf.float32, name='lstm_bw')

            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,cell_bw=lstm_bw_cell, inputs=reduced_dim, dtype=tf.float32, scope='lstm')
            #print(outputs.shape)
            output_fw, output_bw = outputs
            output_concatenated = tf.concat([output_fw, output_bw], axis=2)            
            dropout = tf.nn.dropout(output_concatenated[:,-1,:], keep_prob=self.keep_prob, name='dropout')
            self.outnet = tf.layers.dense(dropout, num_classes,activation=tf.nn.relu, name='output')

    def create_model_stacked_bidirectional(self, num_units_layer1, num_units_layer2, seq_length, num_features, num_classes):
        with self.graph.as_default():

            self.inpnet = tf.placeholder(tf.float32, [None, seq_length ,num_features[1], num_features[2], num_features[3]], name='input')
            self.labels = tf.placeholder(tf.float32, [None, num_classes], name='labels')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            avg_pool = tf.nn.avg_pool3d(self.inpnet, ksize=[1, 1 ,num_features[1], num_features[2], 1], strides=[1, 1, 1, 1, 1], padding='VALID')
            reduced_dim = tf.squeeze(avg_pool, axis=[2, 3])
            print(reduced_dim.shape)
            lstm_fw_cell_1 = tf.nn.rnn_cell.LSTMCell(num_units_layer1)
            lstm_bw_cell_1 = tf.nn.rnn_cell.LSTMCell(num_units_layer1)
            lstm_fw_cell_2 = tf.nn.rnn_cell.LSTMCell(num_units_layer2)
            lstm_bw_cell_2 = tf.nn.rnn_cell.LSTMCell(num_units_layer2)

            cells_fw = [lstm_fw_cell_1, lstm_fw_cell_2]
            cells_bw = [lstm_bw_cell_1, lstm_bw_cell_2]

            lstm_cells_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
            lstm_cells_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)

            (outputs, state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cells_fw,
                                                  cell_bw=lstm_cells_bw,
                                                  inputs=reduced_dim,
                                                  dtype=tf.float32)

            output_fw, output_bw = outputs
            output_concatenated = tf.concat([output_fw, output_bw], axis=2)


            dropout = tf.nn.dropout(output_concatenated[:,-1,:], keep_prob=self.keep_prob, name='2-dropout')
            self.outnet = tf.layers.dense(dropout, num_classes,activation=tf.nn.relu, name='3-dense')

    def create_model_stacked(self, num_units_layer1, num_units_layer2, seq_length, num_features, num_classes):
        with self.graph.as_default():

            self.inpnet = tf.placeholder(tf.float32, [None, seq_length ,num_features[1], num_features[2], num_features[3]], name='input')
            self.labels = tf.placeholder(tf.float32, [None, num_classes], name='labels')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            avg_pool = tf.nn.avg_pool3d(self.inpnet, ksize=[1, 1 ,num_features[1], num_features[2], 1], strides=[1, 1, 1, 1, 1], padding='VALID')
            reduced_dim = tf.squeeze(avg_pool, axis=[2, 3])
            print(reduced_dim.shape)
            lstm_cell_layer1 = tf.nn.rnn_cell.LSTMCell(num_units_layer1)
            lstm_cell_layer2 = tf.nn.rnn_cell.LSTMCell(num_units_layer2)
            stacked_lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_layer1, lstm_cell_layer2])
            #lstm_cell = tf.contrib.rnn.LSTMCell(num_units=1024, dtype=tf.float32, name='1-lstm')
            outputs, states = tf.nn.dynamic_rnn(stacked_lstm_cells, reduced_dim, dtype=tf.float32, scope='1-lstm')
            #print(outputs.shape)
            dropout = tf.nn.dropout(outputs[:,-1,:], keep_prob=self.keep_prob, name='2-dropout')
            self.outnet = tf.layers.dense(dropout, num_classes,activation=tf.nn.relu, name='3-dense')

    def add_train_ops(self, learning_rate):
        with self.graph.as_default():
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outnet, labels=self.labels))
            #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=self.loss)
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss=self.loss)

            
    def add_eval_ops(self):
        with self.graph.as_default():
            self.probabilities = tf.nn.softmax(self.outnet)
            prediction = tf.argmax(self.probabilities, 1)
            correct_prediction = tf.equal(prediction, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
    def train(self, epochs, batch_size, data_train, data_test):
        features_train, labels_train, videos_id_train = data_train
        features_test, labels_test, videos_id_test = data_test
        print(type(features_test))
        print(type(features_train))
        
        num_batches_train = len(features_train) // batch_size
        num_batches_test = len(features_test) // batch_size
 
        with self.graph.as_default():
            #session_config = tf.ConfigProto(device_count={'GPU': 0})
            session_config = tf.ConfigProto()
            session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
            session_config.gpu_options.visible_device_list = "0" 
            saver = tf.train.Saver()
        session = tf.Session(graph=self.graph, config=session_config)

        with session as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):

                sum_train_loss = 0
                sum_train_acc = 0
                for batch_train in range(num_batches_train):
                    batch_frames, batch_labels = features_train[batch_train*batch_size:batch_train*batch_size+batch_size], labels_train[batch_train*batch_size:batch_train*batch_size+batch_size]
                    opt, train_loss, train_acc = sess.run([self.optimizer, self.loss, self.accuracy], feed_dict={self.inpnet:batch_frames, self.labels:batch_labels[:,-1,:], self.keep_prob: 0.6})
                    sum_train_loss += train_loss
                    sum_train_acc += train_acc
                    if batch_train % 16 == 0:
                        print('Batch: {}/{}, Train Loss: {:.2f}, Train accuracy: {:.2f} %'.format(batch_train, num_batches_train, (sum_train_loss/(batch_train+1)), (sum_train_acc/(batch_train+1))*100))
                    
                if(epoch % 1 == 0):    
                    sum_test_acc = 0
                    sum_test_loss = 0
                    for batch_test in tqdm(range(num_batches_test),desc="Processing batches",unit="batch"):
                        batch_frames_test, batch_labels_test = features_test[batch_test*batch_size:batch_test*batch_size+batch_size], labels_test[batch_test*batch_size:batch_test*batch_size+batch_size]
                        test_loss, test_acc = sess.run([self.loss, self.accuracy], feed_dict={self.inpnet:batch_frames_test, self.labels:batch_labels_test[:,-1,:], self.keep_prob: 1.0})
                        sum_test_loss += test_loss
                        sum_test_acc += test_acc
                    print('Epoch: {} Test Loss: {:.2f}, Test accuracy: {:.2f} %'.format(epoch+1, (sum_test_loss/num_batches_test), (sum_test_acc/num_batches_test)*100))
            utils.remove_directory_with_contents('{}/{}/{}'.format(self.tensorflow_savedModels,self.arch,self.name))
            tf.saved_model.simple_save(sess, '{}/{}/{}'.format(self.tensorflow_savedModels,self.arch,self.name), {'input':self.inpnet, 'labels':self.labels, 'keep_prob':self.keep_prob}, {'output':self.outnet})

    def eval(self, batch_size, data_test):
        utils.create_directory('{}/{}/{}/{}/{}'.format(self.curdir,'data','probabilities',self.arch,self.name))
        frames_test, labels_test, videos_id = data_test
        print("Start evaluation")
        print(frames_test.shape)
        print(labels_test.shape)
        print(videos_id.shape)
        num_batches_test = len(frames_test) // batch_size
        with self.graph.as_default():
            session_config = tf.ConfigProto(device_count={'GPU': 0})
            session_config.intra_op_parallelism_threads = 4
            session_config.inter_op_parallelism_threads = 4
            """
            session_config = tf.ConfigProto()
            session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
            session_config.gpu_options.visible_device_list = "0"
            """

        session = tf.Session(graph=self.graph, config=session_config)

        with session as sess:
            meta_graph_def = tf.saved_model.load(sess, tags=['serve'], export_dir='{}/{}/{}'.format(self.tensorflow_savedModels,self.arch,self.name))
            self.inpnet = self.graph.get_tensor_by_name(meta_graph_def.signature_def['serving_default'].inputs['input'].name)
            self.labels = self.graph.get_tensor_by_name(meta_graph_def.signature_def['serving_default'].inputs['labels'].name)
            self.keep_prob = self.graph.get_tensor_by_name(meta_graph_def.signature_def['serving_default'].inputs['keep_prob'].name)
            self.outnet = self.graph.get_tensor_by_name(meta_graph_def.signature_def['serving_default'].outputs['output'].name)
           
            self.add_eval_ops()

            #sess.run(tf.global_variables_initializer())
            sum_test_acc = 0
            total_time = 0
            with open('{}/{}/{}/{}/{}/probabilities_clips_{}.csv'.format(self.curdir,'data','probabilities',self.arch,self.name,self.name), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for batch_test in tqdm(range(num_batches_test),desc="Processing batches",unit="batch"):
                    time_batch = 0
                    start = 0
                    end = 0
                    batch_frames_test, batch_labels_test, batch_videos_id_test = frames_test[batch_test*batch_size:batch_test*batch_size+batch_size], labels_test[batch_test*batch_size:batch_test*batch_size+batch_size],videos_id[batch_test*batch_size:batch_test*batch_size+batch_size]
                    start = time.time()
                    probabilities, test_acc = sess.run([self.probabilities,self.accuracy], feed_dict={self.inpnet:batch_frames_test, self.labels:batch_labels_test[:,-1,:], self.keep_prob: 1.0})
                    end = time.time()
                    time_batch = end - start
                    print("Inference {} time: {:.2f}".format(batch_test+1,time_batch*1000))
              
                    row = np.concatenate((probabilities, batch_labels_test[:,-1,:], batch_videos_id_test[:,-1,:].reshape(-1,1)), axis=1).tolist()[0]
                    writer.writerow(row)  
                    sum_test_acc += test_acc
                    total_time += time_batch

            print('Test accuracy: {:.2f} %'.format((sum_test_acc/num_batches_test)*100))
            #tf.saved_model.simple_save(sess, '{}/{}'.format(self.builtdir,self.dirname), {'input':self.inpnet, 'labels':self.labels}, {'output':self.outnet})
            #saver.save(sess,'{}/{}/{}'.format(self.builtdir,self.dirname,self.filename),write_meta_graph=True)
            #tf.train.write_graph(graph_or_graph_def=sess.graph_def, logdir='{}/{}'.format(self.builtdir,self.dirname), name='{}.pb'.format(self.filename), as_text=False)
        #np.savetxt('datos3.csv', logits, delimiter=',')
