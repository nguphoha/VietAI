"""dnn_tf_sol.py
Solution of deep neural network implementation using tensorflow
Author: Kien Huynh 
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import * 
from dnn_np import test
import pdb


def bat_classification():
    # Load data from file
    # Make sure that bat.dat is in data/
    train_x, train_y, test_x, test_y = get_bat_data()
    train_x, _, test_x = normalize(train_x, train_x, test_x)    

    test_y  = test_y.flatten().astype(np.int32)
    train_y = train_y.flatten().astype(np.int32)
    num_class = (np.unique(train_y)).shape[0]

    # DNN parameters
    hidden_layers = [100, 100, 100] # define 1 NN co 3 layers, moi layer co 100 nodes
    learning_rate = 0.1
    batch_size = 200
    steps = 2000
   
    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[train_x.shape[1]])]


    # Available activition functions
    # https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/activation_functions_
    # tf.nn.relu
    # tf.nn.elu
    # tf.nn.sigmoid
    # tf.nn.tanh
    activation = tf.nn.relu
    
    # [TODO 1.7] Create a neural network and train it using estimator

    # Some available gradient descent optimization algorithms
    # https://www.tensorflow.org/api_guides/python/train#Optimizers
    # tf.train.GradientDescentOptimizer
    # tf.train.AdadeltaOptimizer
    # tf.train.AdagradOptimizer
    # tf.train.AdagradDAOptimizer
    # tf.train.MomentumOptimizer
    # tf.train.AdamOptimizer
    # tf.train.FtrlOptimizer
    # tf.train.ProximalGradientDescentOptimizer
    # tf.train.ProximalAdagradOptimizer
    # tf.train.RMSPropOptimizer
    # Create optimizer

    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.005)
    
    # build a deep neural network
    # https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier
    classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=hidden_layers,
    n_classes=num_class,
    optimizer=optimizer,
    activation_fn=activation,
    model_dir="/tmp/Bat_model") #!! bo sung parameter
    
    # Define the training inputs
    # https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                                    x={"x": train_x},
                                    y=train_y,
                                    batch_size=batch_size,
                                    num_epochs=1, #so lan doc data, neu ko define se chay lap lai forever ??? Khong hieu cho nay, neu de la 1 thi se chay ra ket qua sai
                                    shuffle=False) #khong xao tron thu tu data
    
    # Train model.
    classifier.train(
        input_fn=train_input_fn,
        steps=steps)
    
    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
                                    x={"x": test_x},
                                    y=test_y,
                                    num_epochs=1,
                                    shuffle=False)
    
    # Evaluate accuracy. 
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_x},
      num_epochs=1,
      shuffle=False)
    y_hat = classifier.predict(input_fn=predict_input_fn)
    correctResult = tf.equal(tf.argmax(y_hat,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correctResult, tf.float32))
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    # logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    logdir = "/tmp/retrain_logs/"
    y_hat = list(y_hat)
    y_hat = np.asarray([int(x['classes'][0]) for x in y_hat]) 
    test(y_hat, test_y)


def mnist_classification():
    # Load data from file
    # Make sure that fashion-mnist/*.gz is in data/
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data(1)

    train_x, val_x, test_x = normalize(train_x, val_x, test_x)


    train_y = train_y.flatten().astype(np.int32)
    val_y = val_y.flatten().astype(np.int32)
    test_y = test_y.flatten().astype(np.int32)

    num_class = (np.unique(train_y)).shape[0]

    #pdb.set_trace()

    # DNN parameters
    hidden_layers = [100, 100, 100]
    learning_rate = 0.1
    batch_size = 200
    steps = 500
   
    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[train_x.shape[1]])]


    # Choose activation function
    activation = tf.nn.relu
    
    # Some available gradient descent optimization algorithms 
    # TODO: [YC1.7] Create optimizer
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    
    # build a deep neural network
    classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=hidden_layers,
    n_classes=num_class,
    optimizer= optimizer,
    activation_fn=activation,
    model_dir="/tmp/Mnist_model")
    
    # Define the training inputs
    # https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_x},
        y=train_y,
        num_epochs=1,
        batch_size=batch_size,
        shuffle=False,) #khong xao tron thu tu data
    #so lan doc data, neu ko define se chay lap lai forever ??? Khong hieu cho nay, neu de la 1 thi se chay ra ket qua sai
    # Train model.
    classifier.train(
        input_fn=train_input_fn,
        steps=steps)
    
    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
                                    x={"x": test_x},
                                    y=test_y,
                                    num_epochs=1,
                                    shuffle=False)
    
    # Evaluate accuracy. 
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_x},
      num_epochs=1,
      shuffle=False)
    y_hat = classifier.predict(input_fn=predict_input_fn)
    y_hat = list(y_hat)
    y_hat = np.asarray([int(x['classes'][0]) for x in y_hat]) 
    test(y_hat, test_y)


if __name__ == '__main__':
    np.random.seed(2017) 

    plt.ion()
    bat_classification()
    #mnist_classification()
    pdb.set_trace()
