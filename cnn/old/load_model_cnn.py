import os
import csv
from tqdm import tqdm

import tensorflow.compat.v1 as tf
#import tensorflow as tf
import numpy as np 
import utils


tf.disable_eager_execution()
print(tf.__version__)

class model():
    TARGETS = dict({'ALVEO': 'DPUCAHX8H_ISA2_ELP2', 'ZCU102': 'DPUCZDX8G_ISA0_B4096_MAX_BG2'})

    def __init__(self,load_mode=None,model_architecture=None,model_name=None):
        self.graph = tf.Graph()
        self.load_mode = load_mode
        self.curdir = os.path.abspath(os.path.curdir)
        self.builtdir = '{}/{}'.format(self.curdir,'built')
        self.featuresdir = '{}/{}/{}'.format(self.curdir,'data','ucf101_features')
        self.tensorflow_models = '{}/{}'.format(self.builtdir,'tensorflow_models')
        self.tensorflow_savedModels = '{}/{}'.format(self.builtdir,'tensorflow_savedModels')
        self.frozen_models = '{}/{}'.format(self.builtdir,'frozen_models')
        self.arch = model_architecture
        self.name = model_name
    
    def load_kerasModel(self, size):
        with self.graph.as_default():
            #self.labels = tf.placeholder(tf.float32, [None, num_classes], name='labels')
            if self.arch == 'vgg16':
                baseModel = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(size, size, 3))  
                self.inpnet = self.graph.get_tensor_by_name('input_1:0')
                self.features = self.graph.get_tensor_by_name('block5_pool/MaxPool:0')
            elif self.arch == 'resnet50':
                baseModel = tf.keras.applications.resnet.ResNet50(weights="imagenet", include_top=False, input_shape=(size, size, 3))
                """
                for node in tf.get_default_graph().as_graph_def().node:
                    print(node.name)
                """
                self.inpnet = self.graph.get_tensor_by_name('input_1:0')
                self.features = self.graph.get_tensor_by_name('conv5_block3_out/Relu:0')
            elif self.arch == 'inceptionV3':
                baseModel = tf.keras.applications.inception_v3.InceptionV3(weights="imagenet", include_top=False, input_shape=(size, size, 3))
                
                for node in tf.get_default_graph().as_graph_def().node:
                    print(node.name)
                
                self.inpnet = self.graph.get_tensor_by_name('input_1:0')
                self.features = self.graph.get_tensor_by_name('mixed10/concat:0')
           
            self.saver = tf.train.Saver()
     
    def load_savedModel(self):
        with self.graph.as_default():
            self.meta_graph_def = tf.saved_model.load(sess=self.sess, tags=['serve'], export_dir='{}/{}/{}'.format(self.tensorflow_savedModels,self.arch,self.name))
            self.inpnet = self.graph.get_tensor_by_name(self.meta_graph_def.signature_def['serving_default'].inputs['input'].name)
            self.labels = self.graph.get_tensor_by_name(self.meta_graph_def.signature_def['serving_default'].inputs['labels'].name)
            self.features = self.graph.get_tensor_by_name(self.meta_graph_def.signature_def['serving_default'].outputs['features'].name)


    def create_backend(self,device,size):
        with self.graph.as_default():
            if device == 'cpu':
                #session_config = tf.ConfigProto(allow_soft_placement=True)
                session_config = tf.ConfigProto(device_count={'GPU':0})
            elif device == 'gpu':
                session_config = tf.ConfigProto()
                #session_config.gpu_options.allow_growth = True 
                session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
                session_config.gpu_options.visible_device_list = "0" 
            else:
                print("Not device selected")

            if self.load_mode == 'kerasModel':
                tf.keras.backend.set_session(tf.Session(config=session_config))
                self.load_kerasModel(size)
                self.sess = tf.keras.backend.get_session()
            elif self.load_mode == 'savedModel':
                #tf.keras.backend.set_learning_phase(0)
                self.sess = tf.Session(graph=self.graph, config=session_config)
                self.load_savedModel()
            else:
                print('blablabla')   

         
    def add_frontend_ops_default(self, num_classes):
        with self.graph.as_default():
            if(self.load_mode == 'kerasModel'):
                self.labels = tf.placeholder(tf.float32, [None, num_classes], name='labels')
                self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                avg_pool = tf.nn.avg_pool(self.features, ksize=[1, self.features.shape[1], self.features.shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
                reduced_dim = tf.squeeze(avg_pool, axis=[1, 2])
                with tf.variable_scope("frontend_ops"):
                    kernel = tf.get_variable(name='weights', shape=[reduced_dim.shape[1],2048],initializer=tf.keras.initializers.glorot_uniform())
                    biases = tf.get_variable(name='biases',shape=[2048,],initializer=tf.keras.initializers.glorot_uniform())
                    dense = tf.matmul(reduced_dim, kernel) + biases
                    dropout = tf.nn.dropout(dense, keep_prob=self.keep_prob, name='dropout')
                    kernel1 = tf.get_variable(name='weights1', shape=[2048,num_classes],initializer=tf.keras.initializers.glorot_uniform())
                    biases1 = tf.get_variable(name='biases1',shape=[num_classes,],initializer=tf.keras.initializers.glorot_uniform())
                    dense1 = tf.matmul(dense, kernel1) + biases1
                    self.outnet = tf.nn.dropout(dense1, keep_prob=self.keep_prob, name='dropout1')
                variables_in_scope_frontend = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="frontend_ops")
                print(variables_in_scope_frontend)
                self.init = tf.variables_initializer(variables_in_scope_frontend)
            elif (self.load_mode == 'savedModel'):
                self.keep_prob = self.graph.get_tensor_by_name(self.meta_graph_def.signature_def['serving_default'].inputs['keep_prob'].name)
                self.outnet = self.graph.get_tensor_by_name(self.meta_graph_def.signature_def['serving_default'].outputs['output'].name)
            else:
                print("Failed in add_frontend_ops_default")
  
    def add_train_ops(self, learning_rate):
        with self.graph.as_default():
            with tf.variable_scope("frontend_ops"):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outnet, labels=self.labels))
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss=self.loss)
            variables_in_scope_frontend = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="frontend_ops")
            print(variables_in_scope_frontend)
            self.init = tf.variables_initializer(variables_in_scope_frontend)

    def add_eval_ops(self):
        with self.graph.as_default():
            self.probabilities = tf.nn.softmax(self.outnet)
            prediction = tf.argmax(self.probabilities, 1)
            correct_prediction = tf.equal(prediction, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def tunning(self, num_evals, epochs, batch_size, learning_rate, data):
        data_train, data_test = data
        frames_train, labels_train, videos_id_train = data_train
        frames_test, labels_test, videos_id_test = data_test
        print(frames_train.shape)
        print(labels_train.shape)
        print(frames_test.shape)
        print(labels_test.shape)

        num_batches_train = len(frames_train) // batch_size
        num_batches_test = len(frames_test) // batch_size
        #EVAL_EVERY = num_batches_train
        EVAL_EVERY = num_batches_train // num_evals
        self.add_train_ops(learning_rate)
        self.add_eval_ops()
    
        self.sess.run(self.init)

        results_dict = {}
        for epoch in range(epochs):
            sum_train_loss = 0
            sum_train_acc = 0
            for batch_train in range(num_batches_train):
            #for batch_train in range(3):
                batch_frames, batch_labels = frames_train[batch_train*batch_size:batch_train*batch_size+batch_size], labels_train[batch_train*batch_size:batch_train*batch_size+batch_size]
                opt, train_loss, train_acc = self.sess.run([self.optimizer, self.loss, self.accuracy], feed_dict={self.inpnet:batch_frames, self.labels:batch_labels, self.keep_prob:0.7})
                sum_train_loss += train_loss
                sum_train_acc += train_acc
                print('Batch: {}/{}, Train Loss: {:.2f}, Train accuracy: {:.2f} %'.format(batch_train, num_batches_train, (sum_train_loss/(batch_train+1)), (sum_train_acc/(batch_train+1))*100))
                   
                if ((batch_train != 0) and (batch_train % EVAL_EVERY == 0)):
                    sum_test_acc = 0
                    sum_test_loss = 0
                    for batch_test in tqdm(range(num_batches_test),desc="Processing batches",unit="batch"):
                        batch_frames_test, batch_labels_test = frames_test[batch_test*batch_size:batch_test*batch_size+batch_size], labels_test[batch_test*batch_size:batch_test*batch_size+batch_size]
                        test_loss, test_acc = self.sess.run([self.loss, self.accuracy], feed_dict={self.inpnet:batch_frames_test, self.labels:batch_labels_test, self.keep_prob:1.0}) 
                        sum_test_loss += test_loss
                        sum_test_acc += test_acc
                    print('Epoch: {}, Test Loss: {:.2f}, Test accuracy: {:.2f} %'.format(epoch+1, (sum_test_loss/num_batches_test), (sum_test_acc/num_batches_test)*100))      

    def save(self):
        with self.graph.as_default():
            tf.keras.backend.set_learning_phase(0)
            savedModel_dir = '{}/{}/{}'.format(self.tensorflow_savedModels,self.arch,self.name)
            save_saver_dir = '{}/{}/{}'.format(self.tensorflow_models,self.arch,self.name)
            utils.remove_directory_with_contents(savedModel_dir)
            utils.create_directory(save_saver_dir)
            self.saver.save(sess=self.sess,save_path='{}/{}'.format(save_saver_dir,self.name),write_meta_graph=True)
            tf.saved_model.simple_save(session=self.sess, export_dir='{}'.format(savedModel_dir), inputs={'input':self.inpnet, 'labels':self.labels, 'keep_prob':self.keep_prob}, outputs={'features':self.features,'output':self.outnet})                
            #tf.train.write_graph(graph_or_graph_def=self.sess.graph_def, logdir='{}'.format(save_meta_dir), name='{}.pb'.format(self.name), as_text=False) 

    def eval(self, batch_size, data):
        utils.create_directory('{}/{}/{}/{}/{}'.format(self.curdir,'data','probabilities',self.arch,self.name))
        frames_test, labels_test, videos_id_test = data
        print(frames_test.shape)
        print(labels_test.shape)
        print(videos_id_test.shape)
        num_batches_test = len(frames_test) // batch_size
        self.add_eval_ops()
        sum_test_acc = 0
        with open('{}/{}/{}/{}/{}/probabilities_clips_{}.csv'.format(self.curdir,'data','probabilities',self.arch,self.name,self.name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for batch_test in tqdm(range(num_batches_test),desc="Processing batches",unit="batch"):
                batch_frames_test, batch_labels_test, batch_videos_id_test = frames_test[batch_test*batch_size:batch_test*batch_size+batch_size], labels_test[batch_test*batch_size:batch_test*batch_size+batch_size], videos_id_test[batch_test*batch_size:batch_test*batch_size+batch_size]
                #test_acc = self.sess.run(self.accuracy, feed_dict={self.inpnet:batch_frames_test, self.labels:batch_labels_test, self.keep_prob:1.0})
                probabilities, test_acc = self.sess.run([self.probabilities,self.accuracy], feed_dict={self.inpnet:batch_frames_test, self.labels:batch_labels_test,self.keep_prob:1.0})
                sum_test_acc += test_acc
                row = np.concatenate((probabilities, batch_labels_test, batch_videos_id_test.reshape(-1,1)), axis=1).tolist()[0]
                writer.writerow(row)
        print('Test accuracy: {:.2f} %'.format((sum_test_acc/num_batches_test)*100))

    def infer(self, batch_size, mode, data):
        feature_frames = list()
        feature_labels = list()
        frames, labels, videos_id = data
        num_batches = len(frames) // batch_size
        frames_size = frames.nbytes / (1024**2)
        print(frames.shape)
        print("Tamaño: {:.2f} MB".format(frames_size))
        for batch in tqdm(range(num_batches),desc="Processing batches",unit="batch"):
            #batch_frames, batch_labels = frames[batch*batch_size:batch*batch_size+batch_size], labels[batch*batch_size:batch*batch_size+batch_size]
            features = self.sess.run(self.features, feed_dict={self.inpnet:frames[batch*batch_size:batch*batch_size+batch_size]})
            feature_frames.append(features)
            feature_labels.append(labels[batch*batch_size:batch*batch_size+batch_size])
            
        feature_frames = np.concatenate(feature_frames,axis=0)
        feature_labels = np.concatenate(feature_labels,axis=0)
        feature_frames_size = feature_frames.nbytes / (1024**2)
        print("Tamaño: {:.2f} MB".format(feature_frames_size))
        utils.create_directory('{}/{}/{}'.format(self.featuresdir,self.arch,self.name))
        if mode == 'train':
            np.save('{}/{}/{}/features_train_{}.npy'.format(self.featuresdir,self.arch,self.name,self.name),feature_frames)
            np.save('{}/{}/{}/labels_train_{}.npy'.format(self.featuresdir,self.arch,self.name,self.name), feature_labels)
            np.save('{}/{}/{}/features_id_video_train_{}.npy'.format(self.featuresdir,self.arch,self.name,self.name), videos_id)
        elif mode == 'test':
            np.save('{}/{}/{}/features_test_{}.npy'.format(self.featuresdir,self.arch,self.name,self.name),feature_frames)
            np.save('{}/{}/{}/labels_test_{}.npy'.format(self.featuresdir,self.arch,self.name,self.name), feature_labels)
            np.save('{}/{}/{}/features_id_video_test_{}.npy'.format(self.featuresdir,self.arch,self.name,self.name), videos_id)

        else:
            print("mode error")

      
    def freeze(self):
        print('Starting freeze')
        output_node_names = self.meta_graph_def.signature_def['serving_default'].outputs['features'].name.split(':')[0]
        input_meta_graph = '{}/{}/{}/{}.meta'.format(self.tensorflow_models,self.arch,self.name,self.name)
        #input_meta_graph = '{}/{}/{}/{}.pb'.format(self.tensorflow_models,self.arch,self.name,self.name)
        input_checkpoint = '{}/{}/{}/{}'.format(self.tensorflow_models,self.arch,self.name,self.name)
        output_graph = '{}/{}/{}/{}.pb'.format(self.frozen_models,self.arch,self.name,self.name)
        utils.create_or_clear_directory('{}/{}/{}'.format(self.frozen_models,self.arch,self.name))
        os.system('freeze_graph --input_meta_graph {} --input_checkpoint {} --input_binary true --output_graph {} --output_node_names {}'.format(input_meta_graph,input_checkpoint,output_graph,output_node_names))    
        #os.system('freeze_graph --{} --input_checkpoint {} --input_binary true --output_graph {} --output_node_names {}'.format(input_meta_graph,input_checkpoint,output_graph,output_node_names))            
        print('Finish freeze')


