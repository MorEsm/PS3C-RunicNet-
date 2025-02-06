#!/usr/bin/env python
#!source ~/Miniforge3__02DEC22/bin/activate
#!conda activate tfm1MAX3
#Cell_Challenge 2025
# Pap Smear Cell Classification Challenge (PS3C)
import numpy as np
import pandas as pd
import tensorflow as tf
import keras 
import sys
import os.path
from PIL import Image
from os import path
from os.path import join
from pathlib import Path
home = str(Path.home())


def RunicNet(input_shape, n_classes, filters, residuals):
    class ElementMulti(tf.keras.layers.Layer):
        def call(self, x1, x2):
            return tf.math.multiply(x1,x2)

    # Gated Dconv Feed-Forward Network
    def GDFN(x, filters):
        x1 = BatchNormalization()(x)       
        x2 = Conv2D(filters, 1, strides=1, padding='same')(x1) # 1x1 conv
        x2 = Conv2D(filters, 3, strides=1, padding='same')(x2) # 3x3 conv        
        x1 = Conv2D(filters, 1, strides=1, padding='same')(x1) # 1x1 conv
        x1 = Conv2D(filters, 3, strides=1, padding='same')(x1) # 3x3 conv
        x1 = Activation('gelu')(x1)                            # GeLU activation
        x1 = ElementMulti()([x1, x2])  
        x1 = Conv2D(filters, 1, strides=1, padding='same')(x1) # 1x1 conv
        x  = Add()([x,x1])                                     # Element-wise addition
        return x

    # Enhanced Residual Block
    def ERB(x, filters):
        x2 = Conv2D(filters, 1, strides=1, padding='same')(x)  # 1x1 conv        
        x3 = Conv2D(filters, 3, strides=1, padding='same')(x2) # 3x3 conv        
        x3 = Add()([x2, x3])                                   # Element-wise addition (skip 3x3)        
        x3 = Conv2D(filters, 1, strides=1, padding='same')(x3) # 1x1 conv        
        x  = Add()([x, x3])                                    # Element-wise addition (skip all)
        return x

    # High-Frequuency Attention Block
    def HFAB(x, filters): 
        x1 = Conv2D(filters, 3, strides=1, padding='same')(x) # 3x3 conv
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)                            # ReLU activation
        x1 = ERB(x1, filters)                                  # ERB block
        x1 = Activation('relu')(x1)                            # ReLU activation
        x1 = Conv2D(filters, 3, strides=1, padding='same')(x1) # 3x3 conv
        x1 = BatchNormalization()(x1)                          # Batch normalization
        x1 = Activation('sigmoid')(x1)                         # Sigmoid activation
        x  = ElementMulti()([x, x1])  
        return x
 
    
    input = Input(input_shape, )
    x0 = Conv2D(filters, 1, strides=1, padding='same', activation='relu')(input)   
    x  = ERB(x0, filters=filters)
    x  = HFAB(x, filters=filters)
    x = ERB(x, filters=filters)
    x = HFAB(x, filters=filters)
    x = ERB(x, filters=filters)
    x = HFAB(x, filters=filters)    
    x = ERB(x, filters=filters)
    x = HFAB(x, filters=filters)
    x = Conv2D(filters, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)    
    x = Activation('relu')(x) 
    xs = Add()([x0,x])                                 
    xs1=xs
    xs2=xs

    for i in range(n_residuals):   
        x33 = Conv2D(filters, 3, strides=1, padding='same',kernel_regularizer=regularizers.l2(0.001))(xs)      # 3x3 conv
        x33 = BatchNormalization()(x33)
        x33 = Activation('relu')(x33)

        x11 = Conv2D(filters, 1, strides=1, padding='same',kernel_regularizer=regularizers.l2(0.001))(xs)      # 1x1 conv
        x11 = BatchNormalization()(x11)
        x11 = Activation('relu')(x11)
                
        x3  = Conv2D(filters, 3, strides=1, padding='same',kernel_regularizer=regularizers.l2(0.001))(x11)      # 3x3 conv
        x3  = BatchNormalization()(x3)
        x4  = Activation('relu')(x3)
 
        xs   = Conv2D(filters, 1, strides=1, padding='same',kernel_regularizer=regularizers.l2(0.001))(x4)      # 1x1 conv

        
    for i in range(n_residuals):   
        x33 = Conv2D(filters, 3, strides=1, padding='same',kernel_regularizer=regularizers.l2(0.001))(xs1)      # 3x3 conv
        x33 = BatchNormalization()(x33)
        x33 = Activation('relu')(x33)

        x11 = Conv2D(filters, 1, strides=1, padding='same',kernel_regularizer=regularizers.l2(0.001))(xs1)      # 1x1 conv
        x11 = BatchNormalization()(x11)
        x11 = Activation('relu')(x11)
                
        x3  = Conv2D(filters, 3, strides=1, padding='same',kernel_regularizer=regularizers.l2(0.001))(x11)      # 3x3 conv
        x3  = BatchNormalization()(x3)
        x3  = Activation('relu')(x3)
        xs1 = Add()([xs1,x3,x33])                               

    x = Activation('sigmoid')(xs1)                         
    xs3 = Concatenate()([xs,xs1]) 

    for i in range(n_residuals):   
        x33 = Conv2D(filters*2, 3, strides=1, padding='same',kernel_regularizer=regularizers.l2(0.001))(xs3)      # 3x3 conv
        x33 = BatchNormalization()(x33)
        x33 = Activation('relu')(x33)

        x11 = Conv2D(filters*2, 1, strides=1, padding='same',kernel_regularizer=regularizers.l2(0.001))(xs3)      # 1x1 conv
        x11 = BatchNormalization()(x11)
        x11 = Activation('relu')(x11)
                
        x3  = Conv2D(filters*2, 3, strides=1, padding='same',kernel_regularizer=regularizers.l2(0.001))(x11)      # 3x3 conv
        x3  = BatchNormalization()(x3)
        x3  = Activation('relu')(x3)
        
        xs3 = Add()([xs3,x3,x33])                                 

    xs8 = Conv2D(filters, 1, strides=1, padding='same', activation='relu')(xs3)
    x   = GDFN(xs8, filters=filters, s=1)                                         
    xs9 = Conv2D(filters, 3, strides=1, padding='same', activation='relu')(x)
    xs10 = Add()([xs9,xs8])                               
    x = tf.keras.layers.GlobalAveragePooling2D()(xs10)  
    x = tf.keras.layers.Dense(128, activation='relu')(x)   
    x = tf.keras.layers.Dropout(0.20)(x)
    output = tf.keras.layers.Dense(4, activation='softmax')(x) 
    model = tf.keras.Model(input, output)
    return model

from tensorflow.keras.layers import Input, Activation, Dense, Concatenate, Conv1D, Conv2D, BatchNormalization, Add, Lambda
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import Conv2DTranspose, MaxPool2D
from tensorflow.keras import Model
from tensorflow.keras import backend as K
INPUT_SHAPE = 196, 196, 3
n_filters = 64
n_residuals = 4
N_CLASSES = 4

model = RunicNet(INPUT_SHAPE, n_classes=N_CLASSES, filters=n_filters, residuals=n_residuals)
model.summary()

import scipy.io
from glob import glob
from tensorflow.keras.utils import image_dataset_from_directory

# Define the path to the dataset
image_path = Path('//home/ansatt/morteza/bhome/Cell_Challenge/isbi2025-ps3c-train-dataset/')
#image_path = Path('//home/ansatt/morteza/bhome/Cell_Challenge/isbi2025-ps3c-train-dataset_smaller/')
# Load the dataset and split it
# Define batch size and image size
batch_size = 32
img_size = (196, 196)  # Resize images to 196x196 (or as needed for your model)

# Create training and validation datasets
train_ds_original = image_dataset_from_directory(
    image_path,
    validation_split=0.3,
    subset="training",
    seed=43,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    image_path,
    validation_split=0.3,
    subset="validation",
    seed=43,
    image_size=img_size,
    batch_size=batch_size
)

# Get class names
class_names = train_ds_original.class_names
num_classes = len(class_names)
print("Class Names:", class_names)

# Identify the index of the "Unhealthy" class
unhealthy_class_idx = class_names.index("unhealthy")
print(f"'Unhealthy' class is at index: {unhealthy_class_idx}")

# Define normalization layer
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Normalize the data (first, before batching)
train_ds_original = train_ds_original.map(lambda x, y: (normalization_layer(x), tf.one_hot(tf.cast(y, tf.int32), depth=num_classes)))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), tf.one_hot(tf.cast(y, tf.int32), depth=num_classes)))

# Apply batching to the dataset (for both training and validation)
train_ds = train_ds_original.prefetch(tf.data.experimental.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)





for x_batch, y_batch in train_ds.take(1):
    print(x_batch.shape)  # Should print (batch_size, 196, 196, 3)



class custom_loss(tf.losses.Loss):
    def __init__(self):
        super(custom_loss, self).__init__()

    def call(self, y_true, y_pred, weights=[1.0, 10.0, 1.0]):
        y_pred = y_pred[...,0]
        mse = tf.reduce_mean(tf.losses.MSE(y_true, y_pred), axis=-1)
        ms_ssim = 1 - (tf.image.ssim(y_pred, y_true, 1.0))  # ssim_multiscale not working...
        adv = 0
        return weights[0] * mse + weights[1] * weights[2] * adv

learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.96)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
#optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_schedule)
#optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-5)#learning_rate_schedule)
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5)#learning_rate_schedule)

model.compile(
    optimizer=optimizer,
    #loss=custom_loss(),
    #loss='mean_absolute_error')
    #loss='cosine_similarity')
    #loss='log_cosh')
    #loss='mean_squared_error')   
    #loss='sparse_categorical_crossentropy',
    #loss='binary_crossentropy',
    
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    #loss='categorical_crossentropy', #If labels are one-hot encoded (e.g., [1, 0, 0, 0] for class 0): Use categorical_crossentropy.
#    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)
    #loss='sparse_categorical_crossentropy', metrics=['accuracy'])#If labels are integers (e.g., [0, 1, 2, 3]): Use sparse_categorical_crossentropy.
   # loss='binary_crossentropy', metrics=['accuracy'])
    
# Define the Keras TensorBoard callback.
##logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
##tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# Set class weights (adjust weights based on class distribution)

###class_names = {0: 'unhealthy_1', 1: 'healthy', 2: 'rubbish', 3: 'unhealthy_2'}
class_weights = {0: 1.0,  # Both
                 1: 1.0,  # healthy
                 2: 1.0,  # rubbish
                 3: 5.0}  # unhealthy

# Train the model.
History = model.fit(train_ds,
                    validation_data=val_ds,
                    batch_size=32,
                    epochs=49,
                    class_weight=class_weights,
                    shuffle=False,)#,

 #   callbacks=[tensorboard_callback])

print("-----------------------------------")
print("---- Saving the tained model... ---")
print("-----------------------------------")
model.save('//home//ansatt//morteza//bhome//Cell_Challenge//CellChallenge__model_GDFN_v49.keras')
#model.save('//home//ansatt//morteza//bhome//Cell_Challenge//CellChallenge__model_v01.h5')
#keras.saving.save_model(model, '//home//ansatt//morteza//bhome//Cell_Challenge//CellChallenge__model_v01_.keras')

print("-----------------------------------")
print("-- Saving the model's history... --")
print("-----------------------------------")
import pandas as pd
hist_df = pd.DataFrame(History.history)
hist_csv_file = '//home//ansatt//morteza//bhome//Cell_Challenge//Histroy_GDFN_v49.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
    
    
    
    
#for layer in model.layers[:5]:  # Freeze the first 5 layers
#    layer.trainable = False
#from tensorflow.keras import layers, Model
#model.layers.pop()
# Add a new output layer (e.g., 3-class softmax)
#new_output = layers.Dense(4, activation='softmax')(model.layers[-1].output)
#model = Model(inputs=model.input, outputs=new_output)    
#from tensorflow.keras import layers, Model
#model.compile(
#    optimizer="adam",
#    loss="categorical_crossentropy",  # Use sparse_categorical_crossentropy if no one-hot encoding
#    metrics=["accuracy"]
#)
#model.fit(train_ds, validation_data=val_ds, epochs=5)
#print("-----------------------------------")
#print("---- Saving Re-tained model...  ---")
#print("-----------------------------------")
#model.save('//home//ansatt//morteza//bhome//Cell_Challenge//CellChallenge__model_v46_R.keras')