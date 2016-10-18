HOME = "/fslhome/misaie/tensorflow/convolutional_network"

import sys
sys.path.insert(0, HOME + '/Libs')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

from freeze_graph import freeze_graph
from sklearn.utils import shuffle



mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print("MNIST was uploaded")



# Hyper parameters
IMAGE_SIDE = 32
DEPTH = 3
CLASSES = 4


EPOCHS = 1000
LEARNING_RATE = 0.0001
STEPS_SIZE = 10
FILTERS = 32 
FILTER_SIZE = 5








# Read My Dataset
class Dataset(object):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def get_next_batch(self, batch_size):
        self.features, self.labels = shuffle(self.features, self.labels)
        return self.features[0:batch_size,:], self.labels[0:batch_size,:]
        
        
class Data(object):
    def __init__(self, features, labels):
        
        partition = features.shape[0] / 4
        
        train_features = features[0:partition*3,:]
        train_labels = labels[0:partition*3,:]
        
        test_features = features[partition*3:partition*4,:]
        test_labels = labels[partition*3:partition*4,:]
        
        self.train = Dataset(train_features, train_labels)
        self.test = Dataset(test_features, test_labels)
        

def load_matrix_from_disk(folder, name):
    matrix = np.load(folder + "/" + name + ".npy")
    return matrix

loaded_features = load_matrix_from_disk(HOME + "/MY_data", "features")
loaded_labels = load_matrix_from_disk(HOME + "/MY_data", "labels")

data = Data(loaded_features, loaded_labels)





# Defining some functions that make the defining graph code look cleaner

def weight_variable(shape,name):
    gaussian_matrix = tf.truncated_normal(shape, stddev=0.1)
    weight_matrix = tf.Variable(gaussian_matrix, name=name)
    #weight_matrix = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=shape, name=name)
    return weight_matrix

def bias_variable(shape,name):
    gaussian_vector = tf.truncated_normal(shape, stddev=0.1)
    bias_vector = tf.Variable(gaussian_vector, name=name)
    #bias_vector = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=shape, name=name)
    return bias_vector

def convolution(images_matrix, weight_matrix):
    # [batch, height, width, depth]
    strides = [1,1,1,1]
    convoluted_images_matrix = tf.nn.conv2d(images_matrix, weight_matrix, strides=strides, padding='SAME')
    return convoluted_images_matrix

def max_pool_2x2(images_matrix):
    # [batch, height, width, depth]
    strides = [1,2,2,1]
    ksize = [1,2,2,1] #[1,2,2,1]
    smaller_images_matrix = tf.nn.max_pool(images_matrix, ksize=ksize, strides=strides, padding='SAME')
    return smaller_images_matrix





# Defining the graph (In this case it is convolutional neural network)

# inputs
targets_matrix = tf.placeholder(tf.float32, shape=[None, CLASSES], name="targets_matrix")
features_matrix = tf.placeholder(tf.float32, shape=[None, IMAGE_SIDE * IMAGE_SIDE * DEPTH], name='features_matrix')

# reshaping images as grids instead of vectors for convolution
images_matrix = tf.reshape(features_matrix, [-1, IMAGE_SIDE, IMAGE_SIDE, DEPTH], name="input_node")


with tf.name_scope('convolutional1') as scope:
    # hidden inputs
    weight_matrix_conv1 = weight_variable([FILTER_SIZE, FILTER_SIZE, DEPTH, FILTERS],name="weight_matrix_conv1")
    bias_vector_conv1 = bias_variable([FILTERS], name="bias_vector_conv1")

    # linear operation
    linear_convoluted_matrix_conv1 = convolution(images_matrix, weight_matrix_conv1) + bias_vector_conv1

    # nonlinear operation
    nonlinear_convoluted_matrix_conv1 = tf.nn.relu(linear_convoluted_matrix_conv1)

    # making output smaller
    smaller_matrix_conv1 = max_pool_2x2(nonlinear_convoluted_matrix_conv1)
    
    
    
with tf.name_scope('convolutional2') as scope:
    # hidden inputs
    weight_matrix_conv2 = weight_variable([FILTER_SIZE, FILTER_SIZE, FILTERS, FILTERS * 2],name="weight_matrix_conv2")
    bias_vector_conv2 = bias_variable([FILTERS * 2], name="bias_vector_conv2")

    # linear operation
    linear_convoluted_matrix_conv2 = convolution(smaller_matrix_conv1, weight_matrix_conv2) + bias_vector_conv2
    #linear_convoluted_matrix_conv2 = convolution(nonlinear_convoluted_matrix_conv1, weight_matrix_conv2) + bias_vector_conv2

    # nonlinear operation
    nonlinear_convoluted_matrix_conv2 = tf.nn.relu(linear_convoluted_matrix_conv2)

    # making output smaller
    smaller_matrix_conv2 = max_pool_2x2(nonlinear_convoluted_matrix_conv2)
    
    smaller_matrix_conv4_flat = tf.reshape(smaller_matrix_conv2, [-1, (IMAGE_SIDE / (2 * 2)) * (IMAGE_SIDE / (2 * 2)) * FILTERS * 2])
    
    
#with tf.name_scope('convolutional3') as scope:
    # hidden inputs
#    weight_matrix_conv3 = weight_variable([11, 11, FILTERS * 2, FILTERS * 3],name="weight_matrix_conv3")
#    bias_vector_conv3 = bias_variable([FILTERS * 3], name="bias_vector_conv3")

    # linear operation
#    linear_convoluted_matrix_conv3 = convolution(smaller_matrix_conv2, weight_matrix_conv3) + bias_vector_conv3

    # nonlinear operation
#    nonlinear_convoluted_matrix_conv3 = tf.nn.relu(linear_convoluted_matrix_conv3)

    # making output smaller
#    smaller_matrix_conv3 = max_pool_2x2(nonlinear_convoluted_matrix_conv3)
#    smaller_matrix_conv4_flat = tf.reshape(smaller_matrix_conv3, [-1, (IMAGE_SIDE / (2 * 2)) * (IMAGE_SIDE / (2 * 2)) * FILTERS * 3])
    


#with tf.name_scope('convolutional4') as scope:
    # hidden inputs
#    weight_matrix_conv4 = weight_variable([5, 5, 128, 256], name="weight_matrix_conv4")
#    bias_vector_conv4 = bias_variable([256], name="bias_vector_conv4")

    # linear operation
#    linear_convoluted_matrix_conv4 = convolution(smaller_matrix_conv3, weight_matrix_conv4) + bias_vector_conv4

    # nonlinear operation
#    nonlinear_convoluted_matrix_conv4 = tf.nn.relu(linear_convoluted_matrix_conv4)

    # making output smaller
#    smaller_matrix_conv4 = max_pool_2x2(nonlinear_convoluted_matrix_conv4)

    # making output flat for fully connected layer
#    smaller_matrix_conv4_flat = tf.reshape(smaller_matrix_conv4, [-1, 4 * 4 * 256])



with tf.name_scope('fully_connected_hidden1') as scope:
    # hidden inputs
    weight_matrix_fc1 = weight_variable([(IMAGE_SIDE / (2 * 2)) * (IMAGE_SIDE / (2 * 2)) * FILTERS * 2, 1000], name="weight_matrix_fc1")
    #weight_matrix_fc1 = weight_variable([4 * 4 * 256, 2048], name="weight_matrix_fc1")
    bias_vector_fc1 = bias_variable([1000], name="bias_vector_fc1")

    # linear operation
    linear_hidden_matrix_fc1 = tf.matmul(smaller_matrix_conv4_flat, weight_matrix_fc1) + bias_vector_fc1

    # nonlinear operation
    nonlinear_hidden_matrix_fc1 = tf.nn.relu(linear_hidden_matrix_fc1)
    


with tf.name_scope('fully_connected_output') as scope:
    # hidden inputs
    weight_matrix_fc2 = weight_variable([1000, CLASSES], name="weight_matrix_fc2")
    bias_vector_fc2 = bias_variable([CLASSES], name="bias_vector_fc2")

    # linear operation
    output_matrix_fc2 = tf.matmul(nonlinear_hidden_matrix_fc1, weight_matrix_fc2) + bias_vector_fc2

    # making output probabilities
    probabilities_matrix = tf.nn.softmax(output_matrix_fc2, name = "output_node")


cross_entropy = tf.reduce_mean(-tf.reduce_sum(targets_matrix * tf.log(probabilities_matrix), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(probabilities_matrix,1), tf.argmax(targets_matrix,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




# Summaries
accuracy_summary = tf.scalar_summary('accuracy', accuracy)
loss_summary = tf.scalar_summary('loss function', cross_entropy)




# Training the convolutional neural network
step = 0

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    summaries = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(HOME + '/TensorBoard',graph=sess.graph)
        
    for epoch in range(EPOCHS):
        # This is my data
        features, labels = data.train.get_next_batch(20)
        
        # This is mnist data
        #batch = mnist.train.next_batch(40)
        #features, labels = batch[0], batch[1]
        
        
        
        sess.run(train_step, feed_dict={features_matrix: features, targets_matrix: labels})

        if epoch % STEPS_SIZE == 0:
            summary = sess.run(summaries, feed_dict={features_matrix:features, targets_matrix: labels})
            summary_writer.add_summary(summary, step)
            step = step + 1
            
    summary_writer.close()
            
 # input_graph - TensorFlow 'GraphDef' file to load
    # input_saver - TensorFlow saver file to load
    # input_checkpoint - TensorFlow variables file to load

    # output_graph - Output 'GraphDef' file name
    # input_binary - Whether the input files are in binary format
    # output_node_names - The name of the output nodes, comma separated
    # restore_op_name - The name of the master restore operator
    # filename_tensor_name - The name of the tensor holding the save path
    # clear_devices - Whether to remove device specifications
    # initializer_nodes - comma separated list of initializer nodes to run before freezing

    # Saving the trained network

    path = HOME + "/TrainedModel/"
    checkpoint_weights_filename = "weights"
    checkpoint_graph_filename = "graph.pb"
    trained_graph_filename = "trained_graph.pb"

    checkpoint_weights_path = path + checkpoint_weights_filename
    checkpoint_graph_path = path + checkpoint_graph_filename
    trained_graph_path = path + trained_graph_filename
    saver_path = ""

    as_text = True
    as_binary = not as_text

    
    
    
    # Saving learned weights of the model
    tf.train.Saver().save(sess, checkpoint_weights_path) #, global_step=0, latest_filename="checkpoint_name")

    
    
    # Saving graph definition
    tf.train.write_graph(sess.graph.as_graph_def(), path, checkpoint_graph_filename, as_text)

    
    
    
    # Merging graph definition and learned weights into a trained graph
    input_saver_path = ""
    input_binary = False
    output_node_names = "output_node"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    clear_devices = False

    freeze_graph (
        input_graph = checkpoint_graph_path,
        input_saver = saver_path,
        input_checkpoint = checkpoint_weights_path,
        output_graph = trained_graph_path,

        initializer_nodes = None, #"input_node",
        output_node_names = "fully_connected_output/output_node",

        restore_op_name = "save/restore_all",
        filename_tensor_name = "save/Const:0",

        input_binary = as_binary,
        clear_devices = True
    )

    
      
