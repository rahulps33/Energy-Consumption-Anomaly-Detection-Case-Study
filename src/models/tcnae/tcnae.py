import numpy
from tcn import TCN
import time
import tensorflow
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras import losses
from tensorflow.keras.optimizers import legacy as optimizers_legacy
import pandas

class TCNAE:
    """
    A class used to represent the Temporal Convolutional Autoencoder (TCN-AE).
    """
    
    model = None
    
    def __init__(self,
                 ts_dimension,
                 kernel_size,
                 nb_filters,
                 filters_conv1d,
                 latent_sample_rate,
                 sequence_length,
                 dilations = (1, 2, 4, 8, 16,32),
                 nb_stacks = 1,
                 padding = 'same',
                 dropout_rate = 0.1,
                 activation_conv1d = 'linear',
                 pooler = AveragePooling1D,
                 lr = 0.001,
                 conv_kernel_init = 'glorot_normal',
                 loss = 'logcosh',
                 use_early_stopping = False,
                 error_window_length = 1,
                 
                 verbose = 1
                ):
        """
        Parameters
        ----------
        ts_dimension : int
            The dimension of the time series (default is 1)
        dilations : tuple
            + model (default is (1, 2, 4, 8, 16))
        nb_filters : int
            The number of filters used in the dilated convolutional layers. All dilated conv. layers use the same number of filters (default is 20)
        """
        
        self.ts_dimension = ts_dimension
        self.dilations = dilations
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.filters_conv1d = filters_conv1d
        self.activation_conv1d = activation_conv1d
        self.latent_sample_rate = latent_sample_rate
        self.pooler = pooler
        self.lr = lr
        self.conv_kernel_init = conv_kernel_init
        self.loss = loss
        self.use_early_stopping = use_early_stopping
        self.error_window_length = error_window_length
        self.sequence_length = sequence_length
        
        # build the model
        self.build_model(verbose = verbose)
        
    
    def build_model(self, verbose = 1):
        """Builds the TCN-AE model.

        If the argument `verbose` isn't passed in, the default verbosity level is used.

        Parameters
        ----------
        verbose : str, optional
            The verbosity level (default is 1)
            
        """
        
        tensorflow.keras.backend.clear_session()
        sampling_factor = self.latent_sample_rate
        i = Input(batch_shape=(None, self.sequence_length, self.ts_dimension))
        tcn_enc = TCN(nb_filters=self.nb_filters, kernel_size=self.kernel_size, nb_stacks=self.nb_stacks, dilations=self.dilations, 
                      padding=self.padding, use_skip_connections=True, dropout_rate=self.dropout_rate, return_sequences=True,
                      kernel_initializer=self.conv_kernel_init, name='tcn-enc')(i)
        
        
        # Now, adjust the number of channels...
        enc_flat = Conv1D(filters=self.filters_conv1d, kernel_size = self.kernel_size, activation=self.activation_conv1d, padding=self.padding,name= 'latent')(tcn_enc)
        #print("enc_flat =", enc_flat)
        ## Do some average (max) pooling to get a compressed representation of the time series (e.g. a sequence of length 8)
        enc_pooled = self.pooler(pool_size=sampling_factor, strides=None, padding='valid', data_format='channels_last')(enc_flat)
        # print("enc_pooled =", enc_pooled)
        # If you want, maybe put the pooled values through a non-linear Activation
        enc_out = Activation("linear")(enc_pooled)
        #print("enc_out =", enc_out)
        # Now we should have a short sequence, which we will upsample again and then try to reconstruct the original series
        dec_upsample = UpSampling1D(size=sampling_factor)(enc_out)
        #print("enc_flat =", dec_upsample)
        dec_reconstructed = TCN(nb_filters=self.nb_filters, kernel_size=self.kernel_size, nb_stacks=self.nb_stacks, dilations=self.dilations, 
                                padding=self.padding, use_skip_connections=True, dropout_rate=self.dropout_rate, return_sequences=True,
                                kernel_initializer=self.conv_kernel_init, name='tcn-dec')(dec_upsample)
        #print("dec_reconstructed =", dec_reconstructed)
        # Put the filter-outputs through a dense layer finally, to get the reconstructed signal

        o = Dense(self.ts_dimension, activation='linear')(dec_reconstructed)
        model = Model(inputs=[i], outputs=[o])

        # adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True) // version change
        adam = optimizers_legacy.Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
        #model.add(tf.keras.layers.Flatten())
        model.compile(loss=self.loss, optimizer=adam, metrics=['accuracy'])
        if verbose > 1:
            model.summary()
        self.model = model
        
        print("model.summary():",model.summary())
    
  
      
