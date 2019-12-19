

import numpy as np
from keras.layers import Input, Dense, LeakyReLU
from keras.models import Model
import scipy.io as sio
import h5py
import tensorflow as tf

batch_size = 500
original_dim = 32
epochs = 10
zscore = True

for subNo in range(1,33): #subNo 从1到32
    for latent_dim in range(1,17):
        print('subNo: '+str(subNo)+' latend_dim: '+str(latent_dim))
        # encoder
        eeg_input = Input(shape=(original_dim,))
        dense1 = Dense(latent_dim, activation='sigmoid')(eeg_input)
        encoder = Model(eeg_input, dense1)
        # decoder
        eeg_output = Dense(original_dim)(dense1)

        ae = Model(eeg_input, eeg_output)
        ae.compile(optimizer='adam', loss='mean_squared_error')

        #### train the VAE on normalized (z-score) multi-channel EEG data
        if zscore:
                sub_data_file = h5py.File('D:\\Processed DEAP DATA\\normalize_zscore\\sub'+str(subNo)+'.mat', 'r')
                #sub_data_file = h5py.File('D:\\Processed DEAP DATA\\normalize_zscore_bandpass\\sub' + str(subNo+1) + '.mat', 'r')
                x_train = sub_data_file['zscore_data']
        else:
                sub_data_file = h5py.File('D:\\Processed DEAP DATA\\normalize_minmax\\sub'+str(subNo)+'.mat', 'r')
                #sub_data_file = h5py.File('D:\\Processed DEAP DATA\\normalize_minmax_bandpass\\sub' + str(subNo+1) + '.mat', 'r')
                x_train = sub_data_file['minmax_data']

        x_train = np.transpose(x_train)[:, 0:32]
        x_test = x_train

        ae.fit(x_train, x_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, x_test))

        # build a model to project inputs
        decoded_x = ae.predict(x_train)
        encoded_x = encoder.predict(x_train)


        sio.savemat('D:\\VAE Experiment\\DEAP\\encoded_eegs_1ae\\encoded_eegs_1ae_sub' +
                    str(subNo) + '_latentdim' + str(latent_dim) + '.mat', {'encoded_eegs': encoded_x})

        sio.savemat('D:\\VAE Experiment\\DEAP\\decoded_eegs_1ae\\decoded_eegs_1ae_sub' +
                    str(subNo) + '_latentdim' + str(latent_dim) + '.mat', {'decoded_eegs': decoded_x})