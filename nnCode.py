import splitfolders

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.preprocessing.image import ImageDataGenerator #Data augmentation and preprocessing
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import Xception
from tensorflow.keras import Input
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from tensorflow import keras
from pickle5 import pickle
from sklearn.metrics import classification_report, roc_curve
import matplotlib.pyplot as plt
# %matplotlib inline
# from sklearn.metrics import confusion_matrix

import os
import glob
from pathlib import Path
import pandas as pd


lr_reduction = ReduceLROnPlateau(
                    monitor = 'binary_accuracy',
                    patience = 2,
                    verbose = 1,
                    factor = .3,
                    min_lr = .000001
                )

def create_datagens_for_140k():

    train_path_140k = "real_vs_fake/real-vs-fake/train"
    test_path_140k = "real_vs_fake/real-vs-fake/test"
    classes_140k = ['fake', 'real']

    train_datagen_140k = ImageDataGenerator(
                            rescale=1./255,
                            rotation_range=.2,
                            zoom_range=0.1,
                            horizontal_flip=True,
                            shear_range=0.2,
                            validation_split=0.2)

    training_batch_generator_140k = train_datagen_140k.flow_from_directory(train_path_140k, classes = classes_140k, class_mode = 'binary', target_size = (256,256), color_mode = "rgb", subset = 'training')

    validation_batch_generator_140k = train_datagen_140k.flow_from_directory(train_path_140k, classes = classes_140k, class_mode = 'binary', target_size = (256,256), color_mode = "rgb", subset = 'validation')
    # normalize the image data because CNNs can 
    # converge on an answer quicker when data is [0,1] instead of [0,255].with horizontal flips and 

    test_datagen_140k = ImageDataGenerator(rescale = 1./255)

    # testing_batch[batch number][features or labels][images]
    testing_batch_generator_140k = test_datagen_140k.flow_from_directory(test_path_140k, classes = classes_140k, class_mode = 'binary',
                                                target_size = (256,256), color_mode = "rgb")
    
    return training_batch_generator_140k, validation_batch_generator_140k, testing_batch_generator_140k
    
def create_datagens_for_small():
    
    train_path_small = "faces_split_train_test/train"
    test_path_small = "faces_split_train_test/val"
    classes_small = ['training_fake','training_real']

    train_datagen_small = ImageDataGenerator(
                            rescale=1./255,
                            rotation_range=.2,
                            zoom_range=0.1,
                            horizontal_flip=True,
                            shear_range=0.2,
                            validation_split=0.2)

    training_batch_generator_small = train_datagen_small.flow_from_directory(train_path_small, classes = classes_small, class_mode = 'binary',
                                                target_size = (256,256), color_mode = "rgb", subset = 'training')

    validation_batch_generator_small = train_datagen_small.flow_from_directory(train_path_small, classes = classes_small, class_mode = 'binary',
                                                target_size = (256,256), color_mode = "rgb", subset = 'validation')
    # normalize the image data because CNNs can 
    # converge on an answer quicker when data is [0,1] instead of [0,255].with horizontal flips and 

    test_datagen_small = ImageDataGenerator(rescale = 1./255)

    # testing_batch[batch number][features or labels][images]
    testing_batch_generator_small = test_datagen_small.flow_from_directory(test_path_small, classes = classes_small, class_mode = 'binary',
                                                target_size = (256,256), color_mode = "rgb")
    
    return training_batch_generator_small, validation_batch_generator_small, testing_batch_generator_small

def process_testing_data_140k(testing_batch_generator):
    inputs = []
    outputs = []
    
    for i in range(len(testing_batch_generator) - 400):
#         if i % 50 == 0:
#             print(i)
        inputs.append(testing_batch_generator[i][0][:])
        outputs.append(testing_batch_generator[i][1][:])
    inputs = np.concatenate(inputs)
    outputs = np.concatenate(outputs)
    
    return inputs, outputs

def process_testing_data_small(testing_batch_generator):
    inputs = []
    outputs = []
    
    for i in range(len(testing_batch_generator)):
#         if i % 50 == 0:
#             print(i)
        inputs.append(testing_batch_generator[i][0][:])
        outputs.append(testing_batch_generator[i][1][:])
    inputs = np.concatenate(inputs)
    outputs = np.concatenate(outputs)
    
    return inputs, outputs

def create_simple_model():
    
    simple_model = tf.keras.Sequential()
    simple_model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(256, 256, 3)))
    simple_model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

    simple_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    simple_model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

    simple_model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
    simple_model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

    simple_model.add(layers.Flatten())
    simple_model.add(Dense(32, activation='relu'))
    simple_model.add(Dropout(0.3))
    simple_model.add(layers.Dense(1, activation='sigmoid'))

    simple_model.compile(loss=losses.BinaryCrossentropy(),
                       optimizer=optimizers.Adam(learning_rate=0.00002),
                       metrics=[metrics.BinaryAccuracy(),
                                metrics.TrueNegatives(name="true_negatives"),
                                metrics.Precision(name = "precision"),
                                metrics.Recall(name = "recall")])
    
    return simple_model

def fit_simple_model(model, training_batch_generator, validation_batch_generator):
    
    model_post_fit = model.fit(training_batch_generator, batch_size=100, epochs=5, steps_per_epoch = 100, 
                               validation_data = validation_batch_generator, validation_steps = 100, callbacks = [lr_reduction])
    
    return model_post_fit

def create_transfer_model_for_140k():
    
    xcept_model_140k = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(256, 256, 3)))
    # add layers to the end of the transfer model
    base_140k = xcept_model_140k.output
    base_140k = Flatten()(base_140k)
    base_140k = Dense(256, activation='relu')(base_140k)
    base_140k = Dropout(0.2)(base_140k)
    base_140k = Dense(128, activation='relu')(base_140k)
    base_140k = Dropout(0.2)(base_140k)
    base_140k = Dense(64, activation='relu')(base_140k)
    base_140k = Dropout(0.2)(base_140k)
    base_140k = Dense(1, activation='sigmoid')(base_140k)
    t_model_140k = tf.keras.Model(inputs=xcept_model_140k.input, outputs=base_140k)

    t_model_140k.compile(
            loss=losses.BinaryCrossentropy(),
            optimizer=optimizers.Adam(learning_rate=0.0002),
            metrics=[metrics.BinaryAccuracy(),
                     metrics.TrueNegatives(name="true_negatives"),
                     metrics.Precision(name = "precision"),
                     metrics.Recall(name = "recall")])
    
    return t_model_140k, xcept_model_140k

def create_transfer_model_for_small():
    
    xcept_model_small = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(256, 256, 3)))
    # add layers to the end of the transfer model
    base_small = xcept_model_small.output
    base_small = Flatten()(base_small)
    base_small = Dense(256, activation='relu')(base_small)
    base_small = Dropout(0.2)(base_small)
    base_small = Dense(128, activation='relu')(base_small)
    base_small = Dropout(0.2)(base_small)
    base_small = Dense(64, activation='relu')(base_small)
    base_small = Dropout(0.2)(base_small)
    base_small = Dense(1, activation='sigmoid')(base_small)
    t_model_small = tf.keras.Model(inputs=xcept_model_small.input, outputs=base_small)

    t_model_small.compile(
            loss=losses.BinaryCrossentropy(),
            optimizer=optimizers.Adam(learning_rate=0.0002),
            metrics=[metrics.BinaryAccuracy(),
                     metrics.TrueNegatives(name="true_negatives"),
                     metrics.Precision(name = "precision"),
                     metrics.Recall(name = "recall")])
    
    return t_model_small, xcept_model_small

def fit_transfer_model_140k(t_model_140k, xcept_model_140k, training_batch_generator_140k, validation_batch_generator_140k):
    
    t_model_post_fit_140k = t_model_140k.fit(training_batch_generator_140k, batch_size=100, epochs=3, steps_per_epoch = 100, 
               validation_data = validation_batch_generator_140k, validation_steps = 100, callbacks = [lr_reduction])
    
    print("\nFreezing transfer model layers...")
    for layer in xcept_model_140k.layers:
        layer.trainable = False
        
    t_model_post_fit_140k = t_model_140k.fit(training_batch_generator_140k, batch_size=100, epochs=2, steps_per_epoch = 100, 
               validation_data = validation_batch_generator_140k, validation_steps = 100, callbacks = [lr_reduction])
    
def fit_transfer_model_small(t_model_small, xcept_model_small, training_batch_generator_small, validation_batch_generator_small):
    
    t_model_post_fit_small = t_model_small.fit(training_batch_generator_small, batch_size=100, epochs=3, steps_per_epoch = 100, 
               validation_data = validation_batch_generator_small, validation_steps = 100, callbacks = [lr_reduction])
    
    print("\nFreezing transfer model layers...")
    for layer in xcept_model_small.layers:
        layer.trainable = False
        
    t_model_post_fit_small = t_model_small.fit(training_batch_generator_small, batch_size=100, epochs=2, steps_per_epoch = 100, 
               validation_data = validation_batch_generator_small, validation_steps = 100, callbacks = [lr_reduction])
    
def evaluate_model(model, features_test, labels_test):

    test_set_predictions = model.predict(features_test)
    test_predicted_labels = np.round(test_set_predictions).astype(int).reshape(len(test_set_predictions))

    count = 0
    runTime = 0
    for index in range(len(test_set_predictions)):
        if test_predicted_labels[index] == np.array(labels_test)[index]:
                       count += 1

    print("\nTest Data Accuracy:")
    print(count / len(test_set_predictions))
    
    print("\nRecall Score")
    print(recall_score(labels_test, test_predicted_labels))
    
    print("\nPrecision Score")
    print(precision_score(labels_test, test_predicted_labels))
    
def get_roc_data(model1, model2, featuresTest, labelsTest):
    
    print("Predicting using the simple model...")
    pred_model1 = model1.predict(featuresTest).ravel()
    model1_fpr, model1_tpr, model1_threshold = roc_curve(labelsTest, pred_model1)
    
    print("Predicting using the transfer model...")
    pred_model2 = model2.predict(featuresTest).ravel()
    model2_fpr, model2_tpr, model2_threshold = roc_curve(labelsTest, pred_model2)
    
    return model1_fpr, model2_fpr, model1_tpr, model2_tpr

def plot_roc_curve_comparison_exp_1(model1_fpr, model2_fpr, model1_tpr, model2_tpr):
    
    plt.figure(1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot(model1_fpr, model1_tpr, color="red", label="Simple Model")
    plt.plot(model2_fpr, model2_tpr, color="blue", label="Transfer Model")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Experiment 1 - ROC Curve Comparison\nfor GAN Dataset")
    plt.legend(loc = "upper right")

    plt.rc('font', size=14)          # controls default text sizes
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    plt.rc('legend', fontsize=12)    # legend fontsize
    plt.rc('figure', titlesize=24)  # fontsize of the figure title
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size
    plt.show()
    
def plot_roc_curve_comparison_exp_2(model1_fpr, model2_fpr, model1_tpr, model2_tpr):
    
    plt.figure(1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot(model1_fpr, model1_tpr, color="red", label="Simple Model")
    plt.plot(model2_fpr, model2_tpr, color="blue", label="Transfer Model")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Experiment 2 - ROC Curve Comparison\nfor Photoshop Dataset")
    plt.legend(loc = "lower right")

    plt.rc('font', size=14)          # controls default text sizes
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    plt.rc('legend', fontsize=12)    # legend fontsize
    plt.rc('figure', titlesize=24)  # fontsize of the figure title
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size
    plt.show()
    
def plot_roc_curve_comparison_exp_3(model1_fpr, model2_fpr, model3_fpr, model4_fpr, model1_tpr, model2_tpr, model3_tpr, model4_tpr):
    
    plt.figure(1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot(model1_fpr, model1_tpr, color="red", label="Simple Model - GAN Data")
    plt.plot(model2_fpr, model2_tpr, color="blue", label="Transfer Model - GAN Data")
    plt.plot(model3_fpr, model3_tpr, color="green", label="Simple Model - PS Data")
    plt.plot(model4_fpr, model4_tpr, color="orange", label="Transfer Model - PS Data")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Experiment 3 - ROC Curve Comparison\nBetween Both Datasets")
    plt.legend(loc = "lower right")

    plt.rc('font', size=14)          # controls default text sizes
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    plt.rc('legend', fontsize=12)    # legend fontsize
    plt.rc('figure', titlesize=24)  # fontsize of the figure title
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size
    plt.show()
    
def load_models():
    simple_model_140k = keras.models.load_model('/scratch/brshook/simple_model_140k')
    simple_model_small = keras.models.load_model('/scratch/brshook/simple_model_140k')
    t_model_140k = keras.models.load_model('/scratch/brshook/transfer_model_140k')
    t_model_small = keras.models.load_model('/scratch/brshook/transfer_model_140k')
    
    return simple_model_140k, simple_model_small, t_model_140k, t_model_small

def load_ROC_data(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)