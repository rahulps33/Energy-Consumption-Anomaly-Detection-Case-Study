import logging
from numpy.ma import transpose
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
import torch.nn.functional as F
import torchvision as tv
from scipy.spatial import distance

from models.algorithm_utils import Algorithm, PyTorchUtils
from models.lstmae.lstm_enc_dec_axl import LSTMEDModule

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
from models.algorithm_utils import slide_window,mahalanobis_distance

from models.DGHL import DGHL
from models.DGHL_encoder import DGHL_encoder
from anomaly_detection.utils import de_unfold
from anomaly_detection.plotter import plot_reconstruction_ts, plot_anomaly_scores, plot_reconstruction_occlusion_mask

class LSTMAE_Experiment(Algorithm, PyTorchUtils):
    def __init__(self,  name: str = 'LSTM-ED', num_epochs: int = 10, batch_size: int = 20, lr: float = 1e-3,
                 hidden_size: int = 5, sequence_length: int = 30, train_gaussian_percentage: float = 0.25,
                 n_layers: tuple = (1, 1), use_bias: tuple = (True, True), dropout: tuple = (0, 0),
                 seed: int = None, gpu: int = None, details=True):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.lstmed = None
        self.mean, self.cov = None, None

        self.epoch_loss = {}
        self.reconstructed_op = []
        self.reconstructed_error = []

    def fit(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        indices = np.random.permutation(len(sequences))
        split_point = int(self.train_gaussian_percentage * len(sequences))
        train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
        train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                           sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)

        self.lstmed = LSTMEDModule(X.shape[1], self.hidden_size,
                                   self.n_layers, self.use_bias, self.dropout,
                                   seed=self.seed, gpu=self.gpu)
        self.to_device(self.lstmed)
        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr)

        self.lstmed.train()
        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            for ts_batch in train_loader:
                output = self.lstmed(self.to_var(ts_batch))
                #output,latent_space = self.lstmed(self.to_var(ts_batch))  # For latent Space
                loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
                self.lstmed.zero_grad()
                loss.backward()
                optimizer.step()
            #print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")
            self.epoch_loss[epoch] = loss.item()

        self.lstmed.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.lstmed(self.to_var(ts_batch))
            #output,latent_space = self.lstmed(self.to_var(ts_batch))  # For latent Space
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())


        print('Error Vectors:' )
        print(type(error_vectors))
        
        print(np.shape(error_vectors))

        

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)

        #Additional Info when using cuda
        print('Using device:', self.device)
        print()
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


    def predict(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.lstmed.eval()
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        scores_max = []
        scores_mean = []
        scores_mlog = []
        scores_mahalnobis =[]
        outputs = []
        errors = []
        for idx, ts in enumerate(data_loader):
            
            output = self.lstmed(self.to_var(ts))
            #output,latent_space = self.lstmed(self.to_var(ts))  #change for latent space - add this variable latent space
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            
            diff = (error.view(-1, X.shape[1]).data.cpu().numpy() - self.mean)
            
            score_mlog = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())
            score_max = torch.max(error.view(-1, X.shape[1]).data.cpu(), 1).values
            score_mean = torch.mean(error.view(-1, X.shape[1]).data.cpu(),1)
            score_mahalnobis = np.apply_along_axis(lambda x: 
                    np.matmul(np.matmul(x, np.linalg.inv(self.cov)), x.T) ,1 , diff)
            score_mahalnobis = np.sqrt(score_mahalnobis)
            
            #print(score_mahalnobis)
            scores_mlog.append(score_mlog.reshape(ts.size(0), self.sequence_length))
            scores_max.append(score_max.reshape(ts.size(0), self.sequence_length))
            scores_mean.append(score_mean.reshape(ts.size(0), self.sequence_length))
            scores_mahalnobis.append(score_mahalnobis.reshape(ts.size(0), self.sequence_length))
    
            if self.details:               
                outputs.append(output.cpu().data.numpy())   #outputs.append(output.data.numpy())
                errors.append(error.cpu().data.numpy()) # errors.append(error.data.numpy())
                #embeddings.append(latent_space.cpu().data.numpy())  # For latent Space
        
        # stores seq_len-many scores per timestamp and averages them
        scores_mean = np.concatenate(scores_mean)    
        scores_max = np.concatenate(scores_max)  
        scores_mlog =  np.concatenate(scores_mlog)  
        scores_mahalnobis = np.concatenate(scores_mahalnobis)
        #print('Score', np.shape(scores))

        lattice_mean = np.full((self.sequence_length, data.shape[0]), np.nan)
        
        for i, score in enumerate(scores_mean):
            lattice_mean[i % self.sequence_length, i:i + self.sequence_length] = score
        
        scores_mean = np.nanmean(lattice_mean, axis=0)

        lattice_max = np.full((self.sequence_length, data.shape[0]), np.nan)
        for i, score in enumerate(scores_max):
            lattice_max[i % self.sequence_length, i:i + self.sequence_length] = score
        
        scores_max = np.nanmean(lattice_max, axis=0)

        lattice_mlog = np.full((self.sequence_length, data.shape[0]), np.nan)
        
        for i, score in enumerate(scores_mlog):
            lattice_mlog[i % self.sequence_length, i:i + self.sequence_length] = score
        
        scores_mlog = np.nanmean(lattice_mlog, axis=0)

        lattice_mahal = np.full((self.sequence_length, data.shape[0]), np.nan)

        for i, score in enumerate(scores_mahalnobis):
            lattice_mahal[i % self.sequence_length, i:i + self.sequence_length] = score
        
        scores_mahalnobis = np.nanmean(lattice_mahal, axis=0)
        
        if self.details:
            outputs = np.concatenate(outputs)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, output in enumerate(outputs):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = output
            self.prediction_details.update({'reconstructions_mean': np.nanmean(lattice, axis=0).T})
            self.reconstructed_op = lattice

            errors = np.concatenate(errors)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, error in enumerate(errors):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = error
            self.prediction_details.update({'errors_mean': np.nanmean(lattice, axis=0).T})
            self.reconstructed_error = lattice
        
        #change for latent space
        ###########################################################################################
    #        embeddings = np.concatenate(embeddings)
    #        lattice = np.full((X.shape[0], self.hidden_size),0)
    #        for i, latent_space in enumerate(embeddings):
    #            lattice[i] = latent_space
               
    #        latent_space = lattice
    #   return latent_space,scores
        #############################################################################################

        return scores_mean, scores_max, scores_mlog, scores_mahalnobis


class TCNAE_Experiment:
    
    
    def fit(self,train_X, train_Y, batch_size, epochs=1, verbose = 1,validation_steps=1):
        
        
        
        loss = []
        my_callbacks = None
        if self.use_early_stopping:
            monitor = [EarlyStopping(monitor='val_loss', patience=25, min_delta=1e-4, restore_best_weights=True)]

        keras_verbose = 0
        if verbose > 0:
            print("> Starting the Training...")
            keras_verbose = 2
        start = time.time()
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=25, 
                        verbose=1, mode='auto', restore_best_weights=True)
        history = self.model.fit(train_X, train_Y, 
                            batch_size=batch_size, 
                            epochs=epochs, 
                            validation_split=0.001, 
                            shuffle=True,
                            callbacks= my_callbacks,
                            verbose=keras_verbose)
        loss = history.history['loss']

        if verbose > 0:
            print("> Training Time :", round(time.time() - start), "seconds.")
        return loss

    def predict(self, test_X):
        print("predict entered")
        X_rec =  self.model.predict(test_X)
        
        # do some padding in the end, since not necessarily the whole time series is reconstructed
        X_rec = np.pad(X_rec, ((0,0),(0, test_X.shape[1] - X_rec.shape[1] ), (0,0)), 'constant') 
        

        E_rec = (X_rec - test_X).squeeze()
        E_rec =  E_rec.reshape((-1,self.ts_dimension))
        
        Err = slide_window(pd.DataFrame(E_rec), self.error_window_length, verbose = 0)
        Err = Err.reshape(-1, Err.shape[-1]*Err.shape[-2])

        sel = np.random.choice(range(Err.shape[0]),int(Err.shape[0]*0.98))
        mu = np.mean(Err[sel], axis=0)
        cov = np.cov(Err[sel], rowvar = False)
        sq_mahalanobis = mahalanobis_distance(X=Err[:], cov=cov, mu=mu)
        
        # moving average over mahalanobis distance. Only slightly smooths the signal
        anomaly_score = np.convolve(sq_mahalanobis, np.ones((50,))/50, mode='same')
        anomaly_score_mah = np.sqrt(anomaly_score)
        anomaly_score_L2 = np.sum(E_rec**2, axis = 1)
        anomaly_score_L1= np.sum(E_rec,axis=-1)
        
        
        return anomaly_score_mah,anomaly_score_L2,anomaly_score_L1,X_rec,E_rec
        

    def layers(self,layer_name,test):
      layer_output=self.model.get_layer(layer_name).output
      intermediate_model=tensorflow.keras.models.Model(inputs=self.model.input,outputs=layer_output)
      intermediate_prediction=intermediate_model.predict(test)
      return intermediate_prediction

    def evals(self,test):
      eval_score = self.model.evaluate(test, test, verbose = 0)
      return eval_score
     

class DGHL_Experiment:

    def train_DGHL(mc, train_data, test_data, test_labels, train_mask, test_mask, entities, make_plots, root_dir):
        """
        train_data:
            List of tensors with training data, each shape (n_time, 1, n_features)
        test_data:
            List of tensor with training data, each shape (n_time, 1, n_features)
        test_labels:
            List of arrays with test lables, each (ntime)
        train_mask:
            List of tensors with training mask, each shape (n_time, 1, n_features)
        test_mask:
            List of tensors with test mask, each shape (n_time, 1, n_features)
        entities:
            List of names with entities
        """

        print(pd.Series(mc))
        # --------------------------------------- Random seed --------------------------------------
        np.random.seed(mc['random_seed'])

        # --------------------------------------- Parse paramaters --------------------------------------
        window_size = mc['window_size']
        window_hierarchy = mc['window_hierarchy']
        window_step = mc['window_step']
        n_features = mc['n_features']

        total_window_size = window_size*window_hierarchy

        # --------------------------------------- Data Processing --------------------------------------
        n_entities = len(entities)
        train_data_list = []
        test_data_list = []
        train_mask_list = []
        test_mask_list = []

        # Loop to pre-process each entity 
        for entity in range(0, 1):
            #print(10*'-','entity ', entity, ': ', entities[entity], 10*'-')
            train_data_entity = train_data[entity].copy()
            test_data_entity = test_data[entity].copy()
            train_mask_entity = train_mask[entity].copy()
            test_mask_entity = test_mask[entity].copy()

            #print(len(train_data), train_data_entity.shape, train_mask_entity.shape)

            train_data_entity = train_data_entity.astype(float)
            test_data_entity = test_data_entity.astype(float)
            
            assert train_data_entity.shape == train_mask_entity.shape, 'Train data and Train mask should have equal dimensions'
            assert test_data_entity.shape == test_mask_entity.shape, 'Test data and Test mask should have equal dimensions'
            assert train_data_entity.shape[2] == mc['n_features'], 'Train data should match n_features'
            assert test_data_entity.shape[2] == mc['n_features'], 'Test data should match n_features'

            # --------------------------------------- Data Processing ---------------------------------------
            # Complete first window for test, padding from training data
            padding = total_window_size - (len(test_data_entity) - total_window_size*(len(test_data_entity)//total_window_size))
            test_data_entity = np.vstack([train_data_entity[-padding:], test_data_entity])
            test_mask_entity = np.vstack([train_mask_entity[-padding:], test_mask_entity])

            # Create rolling windows
            train_data_entity = torch.Tensor(train_data_entity).float()
            train_data_entity = train_data_entity.permute(0,2,1)
            train_data_entity = train_data_entity.unfold(dimension=0, size=total_window_size, step=window_step)

            test_data_entity = torch.Tensor(test_data_entity).float()
            test_data_entity = test_data_entity.permute(0,2,1)
            test_data_entity = test_data_entity.unfold(dimension=0, size=total_window_size, step=window_step)

            train_mask_entity = torch.Tensor(train_mask_entity).float()
            train_mask_entity = train_mask_entity.permute(0,2,1)
            train_mask_entity = train_mask_entity.unfold(dimension=0, size=total_window_size, step=window_step)

            test_mask_entity = torch.Tensor(test_mask_entity).float()
            test_mask_entity = test_mask_entity.permute(0,2,1)
            test_mask_entity = test_mask_entity.unfold(dimension=0, size=total_window_size, step=window_step)

            train_data_list.append(train_data_entity)
            test_data_list.append(test_data_entity)
            train_mask_list.append(train_mask_entity)
            test_mask_list.append(test_mask_entity)

        # Append all windows for complete windows data
        train_windows_data = torch.vstack(train_data_list)
        train_windows_mask = torch.vstack(train_mask_list)

        # -------------------------------------------- Instantiate and train Model --------------------------------------------
        print('Training model...')
        model = DGHL(window_size=window_size, window_step=mc['window_step'], window_hierarchy=window_hierarchy,
                     hidden_multiplier=mc['hidden_multiplier'], max_filters=mc['max_filters'],
                     kernel_multiplier=mc['kernel_multiplier'], n_channels=n_features,
                     z_size=mc['z_size'], z_size_up=mc['z_size_up'], z_iters=mc['z_iters'],
                     z_sigma=mc['z_sigma'], z_step_size=mc['z_step_size'],
                     z_with_noise=mc['z_with_noise'], z_persistent=mc['z_persistent'],
                     batch_size=mc['batch_size'], learning_rate=mc['learning_rate'],
                     noise_std=mc['noise_std'],
                     normalize_windows=mc['normalize_windows'],
                     random_seed=mc['random_seed'], device=mc['device'])

        model.fit(X=train_windows_data, mask=train_windows_mask, n_iterations=mc['n_iterations'])

        # -------------------------------------------- Inference on each entity --------------------------------------------
        for entity in range(n_entities):

            rootdir_entity = f'{root_dir}/{entities[entity]}'
            os.makedirs(name=rootdir_entity, exist_ok=True)
            # Plots of reconstruction in train
            print('Reconstructing train...')
            x_train_true, x_train_hat, _, mask_windows = model.predict(X=train_data_list[entity], mask=train_mask_list[entity],
                                                                       z_iters=mc['z_iters_inference'])

            x_train_true, _ = de_unfold(x_windows=x_train_true, mask_windows=mask_windows, window_step=window_step)
            x_train_hat, _ = de_unfold(x_windows=x_train_hat, mask_windows=mask_windows, window_step=window_step)

            x_train_true = np.swapaxes(x_train_true,0,1)
            x_train_hat = np.swapaxes(x_train_hat,0,1)

            if make_plots:
                filename = f'{rootdir_entity}/reconstruction_train.png'
                plot_reconstruction_ts(x=x_train_true, x_hat=x_train_hat, n_features=n_features, filename=filename)
                filename_occlusion_mask = f'{rootdir_entity}/occlusion_mask_train.png'
                plot_reconstruction_occlusion_mask(mask = train_mask_array, x=x_train_true, x_hat=x_train_hat, n_features=n_features, filename=filename_occlusion_mask)
      


            # --------------------------------------- Inference on test and anomaly scores ---------------------------------------
            print('Computing scores on test...')
            score_windows, ts_score, x_windows, x_hat_windows, _, mask_windows = model.anomaly_score(X=test_data_list[entity],
                                                                                                     mask=test_mask_list[entity],
                                                                                                     z_iters=mc['z_iters_inference'])

            # Post-processing
            # Fold windows
            score_windows = score_windows[:,None,None,:]
            score_mask = np.ones(score_windows.shape)
            score, _ = de_unfold(x_windows=score_windows, mask_windows=score_mask, window_step=window_step)
            
            x_test_true, _ = de_unfold(x_windows=x_windows, mask_windows=mask_windows, window_step=window_step)
            x_test_hat, _ = de_unfold(x_windows=x_hat_windows, mask_windows=mask_windows, window_step=window_step)
            x_test_true = np.swapaxes(x_test_true,0,1)
            x_test_hat = np.swapaxes(x_test_hat,0,1)
            score = score.flatten()
            score = score[-len(test_labels[entity]):]

            #print(score)

            if make_plots:
                filename = f'{rootdir_entity}/reconstruction_test.png'
                plot_reconstruction_ts(x=x_test_true, x_hat=x_test_hat, n_features=n_features, filename=filename)

            # Plot scores
            if make_plots:
                filename = f'{rootdir_entity}/anomaly_scores.png'
                plot_anomaly_scores(score=score, labels=test_labels[entity], filename=filename)

            results = {'score': score, 'ts_score':ts_score, 'x_test_true':x_test_true, 'x_test_hat':x_test_hat, 'labels':test_labels,
                        'x_train_true':x_train_true, 'x_train_hat':x_train_hat, 'train_mask': train_mask, 'mc':mc}

            with open(f'{rootdir_entity}/results.p','wb') as f:
                pickle.dump(results, f)


    def train_DGHL_encoder(mc, train_data, test_data, test_labels, train_mask, test_mask, entities, make_plots, root_dir):
        """
        train_data:
            List of tensors with training data, each shape (n_time, 1, n_features)
        test_data:
            List of tensor with training data, each shape (n_time, 1, n_features)
        test_labels:
            List of arrays with test lables, each (ntime)
        train_mask:
            List of tensors with training mask, each shape (n_time, 1, n_features)
        test_mask:
            List of tensors with test mask, each shape (n_time, 1, n_features)
        entities:
            List of names with entities
        """

        print(pd.Series(mc))
        # --------------------------------------- Random seed --------------------------------------
        np.random.seed(mc['random_seed'])

        # --------------------------------------- Parse paramaters --------------------------------------
        window_size = mc['window_size']
        window_hierarchy = mc['window_hierarchy']
        window_step = mc['window_step']
        n_features = mc['n_features']

        total_window_size = window_size*window_hierarchy

        # --------------------------------------- Data Processing --------------------------------------
        n_entities = len(entities)
        train_data_list = []
        test_data_list = []
        train_mask_list = []
        test_mask_list = []

        # Loop to pre-process each entity 
        for entity in range(n_entities):
            #print(10*'-','entity ', entity, ': ', entities[entity], 10*'-')
            train_data_entity = train_data.copy()
            test_data_entity = test_data.copy()
            train_mask_entity = train_mask.copy()
            test_mask_entity = test_mask.copy()

            assert train_data_entity.shape == train_mask_entity.shape, 'Train data and Train mask should have equal dimensions'
            assert test_data_entity.shape == test_mask_entity.shape, 'Test data and Test mask should have equal dimensions'
            assert train_data_entity.shape[2] == mc['n_features'], 'Train data should match n_features'
            assert test_data_entity.shape[2] == mc['n_features'], 'Test data should match n_features'

            # --------------------------------------- Data Processing ---------------------------------------
            # Complete first window for test, padding from training data
            padding = total_window_size - (len(test_data_entity) - total_window_size*(len(test_data_entity)//total_window_size))
            test_data_entity = np.vstack([train_data_entity[-padding:], test_data_entity])
            test_mask_entity = np.vstack([train_mask_entity[-padding:], test_mask_entity])

            # Create rolling windows
            train_data_entity = torch.Tensor(train_data_entity).float()
            train_data_entity = train_data_entity.permute(0,2,1)
            train_data_entity = train_data_entity.unfold(dimension=0, size=total_window_size, step=window_step)

            test_data_entity = torch.Tensor(test_data_entity).float()
            test_data_entity = test_data_entity.permute(0,2,1)
            test_data_entity = test_data_entity.unfold(dimension=0, size=total_window_size, step=window_step)

            train_mask_entity = torch.Tensor(train_mask_entity).float()
            train_mask_entity = train_mask_entity.permute(0,2,1)
            train_mask_entity = train_mask_entity.unfold(dimension=0, size=total_window_size, step=window_step)

            test_mask_entity = torch.Tensor(test_mask_entity).float()
            test_mask_entity = test_mask_entity.permute(0,2,1)
            test_mask_entity = test_mask_entity.unfold(dimension=0, size=total_window_size, step=window_step)

            train_data_list.append(train_data_entity)
            test_data_list.append(test_data_entity)
            train_mask_list.append(train_mask_entity)
            test_mask_list.append(test_mask_entity)

        # Append all windows for complete windows data
        train_windows_data = torch.vstack(train_data_list)
        train_windows_mask = torch.vstack(train_mask_list)

        # -------------------------------------------- Instantiate and train Model --------------------------------------------
        print('Training model...')
        model = DGHL_encoder(window_size=window_size, window_step=mc['window_step'], window_hierarchy=window_hierarchy,
                             hidden_multiplier=mc['hidden_multiplier'], max_filters=mc['max_filters'],
                             kernel_multiplier=mc['kernel_multiplier'], n_channels=n_features,
                             z_size=mc['z_size'], z_size_up=mc['z_size_up'],
                             batch_size=mc['batch_size'], learning_rate=mc['learning_rate'],
                             noise_std=mc['noise_std'],
                             normalize_windows=mc['normalize_windows'],
                             random_seed=mc['random_seed'], device=mc['device'])

        model.fit(X=train_windows_data, mask=train_windows_mask, n_iterations=mc['n_iterations'])

        # -------------------------------------------- Inference on each entity --------------------------------------------
        for entity in range(n_entities):

            rootdir_entity = f'{root_dir}/{entities[entity]}'
            os.makedirs(name=rootdir_entity, exist_ok=True)
            # Plots of reconstruction in train
            print('Reconstructing train...')
            x_train_true, x_train_hat, _, mask_windows = model.predict(X=train_data_list[entity], mask=train_mask_list[entity])

            x_train_true, _ = de_unfold(x_windows=x_train_true, mask_windows=mask_windows, window_step=window_step)
            x_train_hat, _ = de_unfold(x_windows=x_train_hat, mask_windows=mask_windows, window_step=window_step)

            x_train_true = np.swapaxes(x_train_true,0,1)
            x_train_hat = np.swapaxes(x_train_hat,0,1)

            if make_plots:
                filename = f'{rootdir_entity}/reconstruction_train.png'
                plot_reconstruction_ts(x=x_train_true, x_hat=x_train_hat, n_features=n_features, filename=filename)
                filename_occlusion_mask = f'{rootdir_entity}/occlusion_mask_train.png'
                plot_reconstruction_occlusion_mask(mask = train_mask_array, x=x_train_true, x_hat=x_train_hat, n_features=n_features, filename=filename_occlusion_mask)
      

            # --------------------------------------- Inference on test and anomaly scores ---------------------------------------
            print(len(test_data_list), len(test_mask_list))
            print('Computing scores on test...')

            score_windows, ts_score, x_windows, x_hat_windows, _, mask_windows = model.anomaly_score(X=test_data_list[entity],
                                                                                                     mask=test_mask_list[entity])

            # Post-processing
            # Fold windows
            score_windows = score_windows[:,None,None,:]
            print(score_windows)
            score_mask = np.ones(score_windows.shape)
            score, _ = de_unfold(x_windows=score_windows, mask_windows=score_mask, window_step=window_step)
            x_test_true, _ = de_unfold(x_windows=x_windows, mask_windows=mask_windows, window_step=window_step)
            x_test_hat, _ = de_unfold(x_windows=x_hat_windows, mask_windows=mask_windows, window_step=window_step)
            x_test_true = np.swapaxes(x_test_true,0,1)
            x_test_hat = np.swapaxes(x_test_hat,0,1)
            score = score.flatten()
            score = score[-len(test_labels[entity]):]

            if make_plots:
                filename = f'{rootdir_entity}/reconstruction_test.png'
                plot_reconstruction_ts(x=x_test_true, x_hat=x_test_hat, n_features=n_features, filename=filename)

            # Plot scores
            if make_plots:
                filename = f'{rootdir_entity}/anomaly_scores.png'
                plot_anomaly_scores(score=score, labels=test_labels[entity], filename=filename)

            results = {'score': score, 'ts_score':ts_score, 'labels':test_labels, 'mc':mc}
            # results = {'score': score, 'ts_score':ts_score, 'x_test_true':x_test_true, 'x_test_hat':x_test_hat, 'labels':test_labels,
            #             'x_train_true':x_train_true, 'x_train_hat':x_train_hat, 'train_mask': train_mask, 'mc':mc}

            with open(f'{rootdir_entity}/results.p','wb') as f:
                pickle.dump(results, f)
     
