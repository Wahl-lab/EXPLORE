#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# ORT analysis - script network multi classes - developed by Victor Ibañez
# 03.04.2021
# -------------------------------------------------------------------------------------

# -----------------------------------------------------
# import libraries
# -----------------------------------------------------

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from PIL import Image

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)

def network_multi(source_dir, target_dir, project_path, project_name, plot_path):

    # -----------------------------------------------------
    # set directory
    # -----------------------------------------------------

    train_data_dir = os.path.join(target_dir, 'training')
    validation_data_dir = os.path.join(target_dir, 'validation')

    # -----------------------------------------------------
    # set parameters
    # -----------------------------------------------------

    # extract number of training / validation samples
    t = []
    v = []
    for t_file in glob.iglob(train_data_dir + '/**/*.jpg', recursive=True):
        t.append(t_file)
    for v_file in glob.iglob(validation_data_dir + '/**/*.jpg', recursive=True):
        v.append(v_file)
    t_l = len(t)
    v_l = len(v)

    # extract weights
    path = source_dir
    classes = os.listdir(path)

    cnt_list = []
    for cl in classes:
        cnt = 0
        for i in glob.iglob(os.path.join(path,cl) + '/**/*.jpg', recursive=True):
            cnt += 1
        cnt_list.append(cnt)

    w_list = []
    total = sum(cnt_list)
    for i in cnt_list:
        w_list.append((1 / i)*(total)/len(cnt_list)) 

    weights = {}
    for i in range(len(w_list)):
        weights[i] = w_list[i]

    print('weight distribution among classes:',weights)

    # extract image size
    im = Image.open(t[0])
    img_width, img_height = im.size
    
    print('your images are of size: ', img_height, img_width, '3')

    epochs = 50
    batch_size = 15
    if len(classes)==2: 
        nclasses = 1
        loss_type = 'binary_crossentropy' 
        class_type = 'binary' 
        activation_fun = 'sigmoid'
    else:
        nclasses = len(classes)
        loss_type = 'categorical_crossentropy' 
        class_type = 'categorical' 
        activation_fun = 'softmax' 
    nb_train_samples = t_l#*nclasses 
    nb_validation_samples = v_l#*nclasses 
    learning_rate = 0.00015
    dropout_rate = 0.5
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, min_lr=0.000005)
    early = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

    # -----------------------------------------------------
    # set input shape of images
    # -----------------------------------------------------

    input_shape = (img_height, img_width, 3)
        
    # -----------------------------------------------------
    # design the CNN
    # -----------------------------------------------------

    # conv layer 1
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # conv layer 2
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # conv layer 3
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # conv layer 4
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # flatten layer
    model.add(Flatten())

    # dense layer 1
    model.add(Dense(500))
    model.add(Activation('relu'))

    # dense layer 2
    model.add(Dense(500))
    model.add(Activation('relu'))

    # dropout layer
    model.add(Dropout(dropout_rate))

    # output layer
    model.add(Dense(nclasses))
    model.add(Activation(activation_fun))

    opt = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=loss_type,
                  optimizer=opt,
                  metrics=['accuracy'])

    # merge callbacks
    callbacks = [early, reduce_lr]

    #print(model.summary())
    #print("built the model!")

    # -----------------------------------------------------
    # plot a graph of the CNN
    # -----------------------------------------------------

    #plot_model(model, to_file='NORnet.png', show_shapes=True, show_layer_names=True)


    # -----------------------------------------------------
    # read in train and test data from folder structure
    # -----------------------------------------------------

    # uncomment for data augmentation of training data

    train_datagen = ImageDataGenerator(
        rescale=1. / 255#,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True
        )
                            
    # only rescaling for test data

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=class_type)
    print(train_generator)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=class_type)
    print(validation_generator)
    print(validation_generator.class_indices)

    # -----------------------------------------------------
    # train and test the CNN
    # -----------------------------------------------------

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        class_weight = weights,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    # -----------------------------------------------------
    # save the model
    # -----------------------------------------------------

    project = os.path.join(project_path, project_name)
    model.save(os.path.join(project,project_name) + '.h5')

    # -----------------------------------------------------
    # plot training and validation accuracy & loss values
    # -----------------------------------------------------
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    #plt.show()
    plt.savefig(os.path.join(plot_path, 'accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    #plt.show()
    plt.savefig(os.path.join(plot_path, 'loss.png'))
    plt.close()
