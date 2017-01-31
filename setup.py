import os
import pandas as pd
import numpy as np
from scipy.stats import norm

dataDIR = './data' # set the data directory here 


############################################################
# read and clean data 
############################################################
logFile = os.path.join(dataDIR, 'driving_log.csv')
raw_data=pd.read_csv(logFile)
# remove unnecessary info
data = raw_data
data = data.drop(['throttle','speed','brake'], axis=1)
# correct 'faulty' data for three data points :
extreme_angles = raw_data.loc[(raw_data['steering']<-0.75) | (raw_data['steering']>0.75)].index
for i in extreme_angles[0:3] :
    data.loc[i,'steering'] = -0.75
nb_imgs = data.shape[0]
############################################################


############################################################
# assign selection probability for each image
############################################################
nb_steer_bins = 25
steer_bins = np.linspace(-1.,1.001,nb_steer_bins+1)
steer_bin_index = list(range(1,nb_steer_bins+1))
data_bin_index = np.digitize(data['steering'], steer_bins) 
assert len(data_bin_index)==nb_imgs, "Check bins."
# selection probability for each bin
steer_bin_prob = np.array([norm.pdf(x, scale=.8) for x in steer_bins])
steer_bin_prob = {i:steer_bin_prob[i-1]/np.sum(steer_bin_prob) for i in steer_bin_index} 
# selection probability for each image 
img_prob = np.zeros(nb_imgs)
data_steer_bins = {steer_bin : [] for steer_bin in steer_bin_index}
for i in range(nb_imgs) :
    key = data_bin_index[i]
    data_steer_bins[key].append(i)
sum = 0
for index in steer_bin_index :
    bin_size = len(data_steer_bins[index])
    p_bin = steer_bin_prob[index]
    sum += bin_size
    for i in data_steer_bins[index] :
        img_prob[i] = p_bin/bin_size
############################################################   

if __name__ == '__main__':
    print('Number of images in dataset : {}'.format(nb_imgs)) 
    for index in steer_bin_index :
        bin_size = len(data_steer_bins[index])
        p_bin = steer_bin_prob[index]
        print("bin {:2} : {: .2f} < steering angle < {: .2f} : size = {:4} images :  selection probability = {:.3f}"
          .format(index, steer_bins[index-1], steer_bins[index], bin_size, p_bin))