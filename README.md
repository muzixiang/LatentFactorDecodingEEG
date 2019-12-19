# LatentFactorDecodingEEG

These codes are shared with the publication of the "Latent Factor Decoding of Multi-channel EEG for Emotion Recognition through AE-like Neural Networks".

0. The file named 'data_normalization_zscore' is used for preprocessing the multichannel EEGs.

1. The file named 'reconstruction_XXX.py' is used for training the AE-like models. 
   After training, the mined latent factors and the decoded data are saved into pre-defined local file path.

2. The file named 'get_ica_encoded.m' and 'get_pca_encoded.m' are used for get the encoded data of ICA and PCA based methods.
   The mined latent factors and the decoded data are saved into pre-defined local file path.
   
3. The file named 'performance_XXX_Rvalue_encoded_decoded.m' is used for calcluate and measure the reconstruction performance. 
   The obtained results are stored in the folder 'calclulated_reconstruction_performance'
   
4. The file named 'plot_Rvalue_latentdims_each_method' and 'plot_Rvalue_latentdims_each_subject' are used for 
   plotting the reconstruction performance.

5. The file named 'prediction_LSTM-RNN' is used to constructing training sequences from the mined latent factors, training the LSTM based
   sequence modeling method and conducting emotion recognition.
