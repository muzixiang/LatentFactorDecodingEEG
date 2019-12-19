

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, LeakyReLU
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, mean_squared_error
from keras.utils import plot_model
from keras import backend as K
from keras import optimizers

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import h5py
import scipy.io as sio

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


batch_size = 512
original_dim = 32
epochs = 10
subNum = 32
zscore = True


for subNo in range(25,33): #subNo 从1到32
    for latent_dim in range(1,17): #latent_dim 从2到11/16/
        #### train the VAE on normalized (z-score) multi-channel EEG data
        if zscore:
            sub_data_file = h5py.File('D:\\Processed DEAP DATA\\normalize_zscore\\sub' + str(subNo) + '.mat', 'r')
            x_train = sub_data_file['zscore_data']
        else:
            sub_data_file = h5py.File('D:\\Processed DEAP DATA\\normalize_minmax\\sub' + str(subNo) + '.mat', 'r')
            x_train = sub_data_file['minmax_data']

        x_train = np.transpose(x_train)[:, 0:32]
        x_test = x_train

        # VAE model = encoder + decoder

        # build encoder model
        inputs = Input(shape=(original_dim, ), name='encoder_input')
        h = Dense(128, activation=LeakyReLU(alpha=0.3))(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(h)
        z_log_var = Dense(latent_dim, name='z_log_var')(h)
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        #encoder.summary()
        #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        h_decoded = Dense(128, activation=LeakyReLU(alpha=0.3))(latent_inputs)
        inputs_decoded = Dense(original_dim)(h_decoded)
        # instantiate decoder model
        decoder = Model(latent_inputs, inputs_decoded, name='decoder')
        #decoder.summary()
        #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')

        # VAE loss = mse_loss or xent_loss + kl_loss
        #reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss = mean_squared_error(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)

        rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        vae.compile(optimizer=rmsprop)
        #vae.summary()
        #plot_model(vae, to_file='vae_mlp.png', show_shapes=True)

        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        # save the model
        # vae.save_weights('vae_mlp_mnist.h5')

        # build a model to project inputs
        encoded_x_zmean = encoder.predict(x_train)[0]
        encoded_x_z = encoder.predict(x_train)[2]
     
        sio.savemat('D:\\VAE Experiment\\DEAP\\encoded_eegs_2vae\\encoded_eegs_2vae_zmean_sub' +
                    str(subNo) + '_latentdim' + str(latent_dim) + '.mat',
                    {'encoded_eegs_zmean': encoded_x_zmean})
        sio.savemat('D:\\VAE Experiment\\DEAP\\encoded_eegs_2vae\\encoded_eegs_2vae_z_sub' +
                    str(subNo) + '_latentdim' + str(latent_dim) + '.mat',
                    {'encoded_eegs_z': encoded_x_z})

        decoded_x = vae.predict(x_train)
        sio.savemat('D:\\VAE Experiment\\DEAP\\decoded_eegs_2vae\\decoded_eegs_2vae_sub' +
                    str(subNo) + '_latentdim' + str(latent_dim) + '.mat',
                    {'decoded_eegs': decoded_x})