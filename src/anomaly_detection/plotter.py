import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pd.options.plotting.backend = "plotly"


def lstmae_plotter(model_details, test, exp_id, model_name, building):

  test_check = test.loc[:, test.columns != 'label']
  test_check.drop(columns=['anomaly_score', 'std_anomaly_score'], inplace = True)
  reconstructed_op = pd.DataFrame(model_details['prediction_details']['reconstructions_mean'].T, index=test_check.index, columns=test_check.columns)
  reconstructed_error = pd.DataFrame(model_details['prediction_details']['errors_mean'].T, index=test_check.index, columns=test_check.columns)
  
  # Reconstruction of features per day
  reconstructed_op = reconstructed_op.resample("D").max()
  reconstructed_error = reconstructed_error.resample("D").max()
  test_check = test_check.resample("D").max()

  for col in test_check.columns:
    fig = make_subplots(rows=3, cols=1, subplot_titles=(col,  'Reconstructed Output', 'Reconstructed Error'))
    fig.append_trace(go.Scatter(x=test_check.index, y=test_check[col], name=col), row=1, col=1)
    fig.append_trace(go.Scatter(x=test_check.index, y=reconstructed_op[col], name='Reconstructed Output'), row=2, col=1)
    fig.append_trace(go.Scatter(x=test_check.index, y=reconstructed_error[col], name='Reconstructed Error'), row=3, col=1)
    fig.update_layout(title=f'Reconstruction of {col}')
    #fig.show()
    fig.write_image(f"outputs/{exp_id}_{model_name}_{building}/Reconstructed Figures/Reconstruction of {col}.pdf")

def tcnae_plotter(model_details, exp_id, model_name,build):
    

    # plot for epoch vs loss
    
    epoch_loss = pd.DataFrame(model_details['epoch_loss'])
    fig = epoch_loss.plot()
    fig.update_layout(title="Epoch vs Loss", xaxis_title="No. of Epochs", yaxis_title="Loss")
    #fig.show()
    plt.savefig(f'outputs/{exp_id}_{model_name}_{build}/Evaluation/epoch vs loss.pdf')

    
    #fetch test
    x = model_details['test']
    shape = x.shape
    x = np.reshape(x, (shape[0]*shape[1], shape[2]))
    length = len(x)
    time = model_details['time']
    time.reset_index(inplace = True,drop = True)
    time = time[:length] 
    y= pd.DataFrame(x[:, :-1], columns= model_details['col_name'],index = time )

    #fetch reconstructed
    x = model_details['reconstructed_op']
    shape = x.shape
    x = np.reshape(x, (shape[0]*shape[1], shape[2]))
    y2= pd.DataFrame(x, columns= model_details['col_name'], index = time)

    #fetch error
    x = model_details['reconstructed_error']
    shape = x.shape
    y3= pd.DataFrame(x, columns= model_details['col_name'], index = time)


    for col in model_details['col_name']:
        fig = make_subplots(rows=3, cols=1, subplot_titles=(col,  'Reconstructed Output', 'Reconstructed Error'))
        fig.append_trace(go.Scatter(x=y.index, y=y[col], name=col), row=1, col=1)
        fig.append_trace(go.Scatter(x=y2.index, y=y2[col], name='Reconstructed Output'), row=2, col=1)
        fig.append_trace(go.Scatter(x=y3.index, y=y3[col], name='Reconstructed Error'), row=3, col=1)
        fig.update_layout(title=f'Reconstruction of {col}')
        fig.show()
        fig.write_image(f"outputs/{exp_id}_{model_name}_{build}/Reconstructed Figures/Reconstruction of {col}.pdf")

def plot_reconstruction_ts(x, x_hat, n_features, filename):
    n_features = n_features - 5
    if n_features>1:
        fig, ax = plt.subplots(n_features, 1, figsize=(15, n_features))
        for i in range(n_features):
            ax[i].plot(x[i])
            ax[i].plot(x_hat[i])
            ax[i].grid()
    else:
        fig = plt.figure(figsize=(15,6))
        plt.plot(x[0])
        plt.plot(x_hat[0])
        plt.grid()
        
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close('all')

def plot_reconstruction_occlusion_mask(mask, x, x_hat, n_features, filename):
    n_features = n_features - 5
    if n_features>1:
        fig, ax = plt.subplots(n_features, 1, figsize=(15, n_features))
        for i in range(n_features):
            ax[i].plot(x[i])
            ax[i].plot(x_hat[i], alpha=0.8)
            for j in range(len(mask[i])):
              if mask[i][j]==0 and mask[i][j-1]==0:
                ax[i].axvspan(j-1, j, color='gray', alpha=0.4, lw=0)
            ax[i].grid()
    else:
        fig = plt.figure(figsize=(15,6))
        plt.plot(x[0])
        plt.plot(x_hat[0])
        for j in range(mask[1]):
              if mask[1][j]==0 and mask[1][j-1]==0:
                plt.axvspan(j-1, j, color='gray', alpha=0.4, lw=0)
        plt.grid()
        
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close('all')


def plot_reconstruction_prob_ts(x, x_mu, x_sigma, n_features, filename):
    n_features = n_features - 5
    fig, ax = plt.subplots(n_features, 1, figsize=(15, n_features))
    for i in range(n_features):
        ax[i].plot(x[i])
        ax[i].plot(x_mu[i,:], color='black')
        ax[i].fill_between(range(len(x_mu[i,:])), x_mu[i,:] - x_sigma[i,:], x_mu[i,:] + x_sigma[i,:], color='blue', alpha=0.25)
        ax[i].grid()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close('all')

def plot_anomaly_scores(score, labels, filename):
    fig, ax = plt.subplots(2, 1, figsize=(15,10))
    ax[0].plot(score, label='anomaly score')
    ax[1].plot(labels, label='label', c='orange')
    ax[0].grid()
    ax[1].grid()
    ax[0].legend()
    ax[1].legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close('all')
