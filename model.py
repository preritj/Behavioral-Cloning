import setup # for loading data, etc
from setup import * 

import preprocess  # for image pre-processing
from preprocess import *

from glob import glob
import numpy as np
import random, cv2, os
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

 


cam_positions = ['left','center','right']
# steering angle shifts for different cam positions : 
delta_steer = {'left':0.4, 'center':0., 'right':-0.4} 


############################################################
# read image from data and perform various transformations
############################################################
# class for generating images from files 
class generate_image :
    def __init__(self, img_index, cam_pos='center') :
        self.img_data = data.iloc[img_index]
        self.steer = self.img_data['steering']
        self.set_camera(cam_pos)
    def reset(self) :
        self.set_camera('center')
    # camera selection
    def set_camera(self, cam_pos) :
        self.cam_pos = cam_pos
        if cam_pos=='random' :
            self.cam_pos = random.choice(cam_positions)
        file_name = os.path.join(dataDIR,self.img_data[self.cam_pos].strip())
        img = cv2.imread(file_name)
        self.image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.new_steer = self.steer + delta_steer[self.cam_pos]
    # flip image about vertical axis
    def flip(self) :
        self.image = cv2.flip(self.image,1)
        self.new_steer = -self.new_steer
    # viewpoint transformations - shift and rotation
    def viewpoint_transform(self, shift=0.4, rotation=0.4) :
        delta1 = 70*shift/delta_steer['left'] # calibration
        delta2 = 70*rotation/delta_steer['left'] # calibration
        h,w,_ = self.image.shape
        pts1 = np.float32([[w/2-30,h/3],[w/2+30,h/3],[w/2-80,h],[w/2+80,h]])
        pts2 = np.float32([[w/2-30+delta2,h/3],[w/2+30+delta2,h/3],
                           [w/2-80-delta1,h],[w/2+80-delta1,h]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        self.image = cv2.warpPerspective(self.image,M,(w,h))
        self.new_steer += -shift + rotation
############################################################

input_shape = (66,200,3) # shape of the image after pre-processing 

############################################################
# python generator for on-the-fly data augmentation
############################################################
# python generator 
class ImageGenerator :
    
    def __init__(self, data_indices, augment=False, randomize=False) :
        self.data_indices = data_indices
        self.augment=augment
        self.randomize = randomize
        self.start_index = None if self.randomize else 0
    
    # this is the generator required by Keras 
    def flow_from_dir(self, batch_size=64) :
        X_batch = np.zeros((batch_size, *input_shape))
        y_batch = np.zeros(batch_size)
        
        # infinite loop for generator 
        while True : 
            batch_indices = None
            if self.randomize :
                prob =  img_prob[self.data_indices]/np.sum(img_prob[self.data_indices])
                batch_indices = np.random.choice(self.data_indices, batch_size, p=prob) 
                #batch_indices = shuffle(self.data_indices, n_samples=batch_size)
            else :
                batch_indices = self.data_indices[self.start_index:self.start_index+batch_size]
                self.start_index += batch_size
                
                # start looping if sample exhausted
                if len(batch_indices) < batch_size :
                    diff = batch_size-len(batch_indices)
                    batch_indices += self.data_indices[0:diff]
                    self.start_index=diff          
            
            # generate images
            for i,img_index in enumerate(batch_indices) :
                img = None 
                if self.augment :
                    # generate image with random camera position
                    img = generate_image(img_index, cam_pos='random')    
                    # flip image with probability 0.5
                    if (random.choice([0,1])==0) :
                        img.flip()
                    # apply random perspective transformation 
                    img.viewpoint_transform(
                        shift=random.uniform(delta_steer['right']/2,delta_steer['left']/2),
                        rotation=random.uniform(delta_steer['right']/2,delta_steer['left']/2))  
                else :
                    img=generate_image(img_index)
                X_batch[i,:,:,:] = preprocess(img.image)
                y_batch[i] = img.new_steer
            yield (X_batch, y_batch)
############################################################
            

############################################################
# CNN architecture
############################################################
from keras.layers import Convolution2D, Dense, Activation, Flatten, Dropout
from keras.models import Sequential, load_model

# for convolutional layers, NVIDIA architecture is used
def nvidia_base() :
    model = Sequential()
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu', input_shape=input_shape))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Flatten())
    return model

# add FC layers to NVIDIA conv layers
model = nvidia_base()
model.add(Dense(250,activation='relu'))
model.add(Dense(1))
model.summary()
############################################################


############################################################
# Model training
############################################################
BATCH_SIZE=64
EPOCHS=3
nb_samples = int(8000/BATCH_SIZE)*BATCH_SIZE # samples per epoch
Validate = False # whether to use validation set

nb_valid = None  # number of validation samples
training_data, validation_data=None, None

if Validate : 
    nb_valid = int(1000/BATCH_SIZE)*BATCH_SIZE  
    train_indices, valid_indices = train_test_split(list(range(nb_imgs)), test_size=nb_valid, random_state=0)
    train_generator = ImageGenerator(train_indices, augment=True, randomize=True)
    valid_generator = ImageGenerator(valid_indices)
    nb_train = len(train_indices)
    print('Number of images in training set before augmentation = {}'.format(nb_train))
    print('Number of images in validation set  = {}'.format(nb_valid))
    print('Number of images in training set before augmentation = {}'.format(nb_train))
    print('Number of images in validation set  = {}'.format(nb_valid))
    training_data=train_generator.flow_from_dir(batch_size=BATCH_SIZE)
    validation_data=valid_generator.flow_from_dir(batch_size=BATCH_SIZE)
else :
    train_generator = ImageGenerator(list(range(nb_imgs)), augment=True, randomize=True)
    training_data=train_generator.flow_from_dir(batch_size=BATCH_SIZE)

## train model
print('Training ...')
model.compile(loss='mse', optimizer='adam')
model.fit_generator(training_data, samples_per_epoch=nb_samples, nb_epoch=EPOCHS, 
                   validation_data=validation_data, nb_val_samples=nb_valid)

## save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
############################################################