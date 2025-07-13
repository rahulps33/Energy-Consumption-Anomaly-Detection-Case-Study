import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
pd.options.plotting.backend = "plotly"

from anomaly_detection.plotter import lstmae_plotter
from anomaly_detection.plotter import tcnae_plotter
#from anomaly_detection.plotter import dghl_plotter

from anomaly_detection.utils import f1_score, best_f1_linspace, normalize_scores
from anomaly_detection.utils_data import get_random_occlusion_mask, dghl_normalization

from sklearn.manifold import TSNE
import plotly.subplots as sp
import plotly.graph_objs as go
from sklearn import metrics

BUILDINGS = ['OH12','OH14','Chemie','Großtagespflege','Kita_hokida', 'EF42','EF40', 'EF40a','Erich_brost', 'Chemie_concat', 'Chemie_single','Office_concat', 'Office_single']

PROJECT_DIR = Path.cwd()

def lstmae_evaluator(exp_id, model_name, test_data, building):

  test = test_data
  Path(PROJECT_DIR, "outputs", rf"{exp_id}_{model_name}_{building}", "Evaluation").mkdir(parents=True, exist_ok=True)
  Path(PROJECT_DIR, "outputs", rf"{exp_id}_{model_name}_{building}", "Reconstructed Figures").mkdir(parents=True, exist_ok=True)
  
  with open(rf"outputs/{exp_id}_{model_name}_{building}/anomaly_score.pkl","rb") as op:
      X_test_pred = pickle.load(op)
  with open(rf"outputs/{exp_id}_{model_name}_{building}/pred_details.pkl","rb") as op:
      model_details = pickle.load(op) 

  # plot for epoch vs loss
  lists = sorted(model_details['epoch_loss'].items())
  x, y = zip(*lists)
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=x, y=y, name='Epoch vs Loss'))
  fig.update_layout(title="Epoch vs Loss", xaxis_title="No. of Epochs", yaxis_title="MSE Loss")
  #fig.show()
  fig.write_image(f"outputs/{exp_id}_{model_name}_{building}/Evaluation/Epoch vs Loss.pdf")

    
  for key,values in X_test_pred.items():

    scaler = MinMaxScaler()
    test['anomaly_score'] =  values
    test['std_anomaly_score'] = scaler.fit_transform(test[['anomaly_score']])
    # threshold = np.percentile(test['std_anomaly_score'], 90)
    # test['anomaly'] = np.where(test['std_anomaly_score'] >= threshold , 1, 0)

    # Resample test data by day
    test_per_day = test.resample("D").max()
    threshold = np.percentile(test_per_day['std_anomaly_score'], 90)
    test_per_day['anomaly'] = np.where(test_per_day['std_anomaly_score'] >= threshold , 1, 0)
    
    # Plot for ROC curve
    fpr, tpr, thresholds = roc_curve(test_per_day['label'], test_per_day['std_anomaly_score'])
    fig = px.area( x=fpr, y=tpr,
        title=f'ROC Curve with Scoring function : {key} <br> (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=400, height=400)
    fig.add_shape(type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.update_layout(font=dict(size=10))
    #fig.show()
    fig.write_image(f"outputs/{exp_id}_{model_name}_{building}/Evaluation/ROC Curve with Scoring function :{key}.pdf")

    # Plot for F1 Score
    target_names = ['Normal', 'Anomaly']
    report = classification_report(test_per_day['label'], test_per_day['anomaly'], target_names = target_names, output_dict=True)
    df = pd.DataFrame(report)[['Normal','Anomaly']].transpose()

    # Ploting the classification report
    fig = px.imshow(df[['precision', 'recall', 'f1-score']], 
                    color_continuous_scale='RdBu_r', text_auto=True)
    fig.update_layout(title=f'Classification Report <br> Scoring function : {key}', xaxis_title="Metrics",
                      yaxis_title="Classes",
                      xaxis=dict(side='bottom'))
    #fig.show()
    fig.write_image(f"outputs/{exp_id}_{model_name}_{building}/Evaluation/Classification Report with Scoring function :{key}.pdf")

    # Ploting confusion matrix
    confusion_matrix = metrics.confusion_matrix(test_per_day['label'], test_per_day['anomaly'])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Normal', 'Anomaly'])
    cm_display.plot()
    #plt.show()
    plt.savefig(f'outputs/{exp_id}_{model_name}_{building}/Evaluation/Confusion Matrix with Scoring function : {key}.pdf')  

    # plot for anomaly score
    fig = make_subplots(rows=2, cols=1)
    fig.append_trace(go.Scatter(x=test_per_day.index, y=test_per_day['label'], name='label'), row=1, col=1)
    fig.append_trace(go.Scatter(x=test_per_day.index, y=test_per_day['anomaly'], name='predicted label'), row=2, col=1)
    fig.update_layout(height=400, width=700, showlegend=True, title=f'True Label vs Predicted Anomaly <br> Scoring function :{key}')
    #fig.show()
    fig.write_image(f"outputs/{exp_id}_{model_name}_{building}/Evaluation/True Label vs Predicted Anomal with Scoring function :{key}.pdf")

    lstmae_plotter(model_details, test, exp_id, model_name, building)


def tcnae_evaluator(exp_id, model_name,build):
    
    with open(rf"outputs/{exp_id}_{model_name}_{build}/pred_details_{build}.pkl","rb") as op:
        model_details = pickle.load(op)
    print("evaluator")

    Path(PROJECT_DIR, "outputs", rf"{exp_id}_{model_name}_{build}", "Evaluation").mkdir(parents=True, exist_ok=True)
    Path(PROJECT_DIR, "outputs", rf"{exp_id}_{model_name}_{build}", "Reconstructed Figures").mkdir(parents=True, exist_ok=True)
        
    keys_to_include = ['anomaly_score_mahanabolis', 'anomaly_score_L1', 'anomaly_score_L2']

    for key in keys_to_include:
        if key in model_details:
            print(key)
            
          #anomaly score and evaluation


            score = model_details[key] #fetching anomaly score from the dictionary
            


            x = model_details["test"] #fetching original test data from the dictionary
            shape = x.shape
            label =  x[:, :,shape[2]-1] # slicing out only the labels
            label=np.reshape(label, -1)
            length = len(label) 

            time = model_details['time'] #fetching time index values from the dictionary

              
            time.reset_index(inplace = True,drop = True)
            time = time[:length] 

            score =np.reshape(score, (-1,1)) #reshaping anomaly score


            # transform data
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(score)



            plot = pd.DataFrame() #merging all required data into a dataframe



            plot['label'] = label
            plot['time'] =  pd.to_datetime(time)
            plot['scaled'] = pd.DataFrame(scaled)
            plot['anomaly']= 0
            plot = plot.set_index('time')
            grouped = plot.resample('D').mean()
            threshold = np.percentile(grouped['scaled'],85)
            print(threshold)
            plot[['label', 'anomaly']] = plot[['label', 'anomaly']].astype(int)




            grouped.loc[grouped['scaled'] <= threshold, 'anomaly'] = 0
            grouped.loc[grouped['scaled'] > threshold, 'anomaly'] = 1
            grouped[['label', 'anomaly']] = grouped[['label', 'anomaly']].astype(int)




            fig = sp.make_subplots(rows=2, cols=1)

            fig.add_trace(go.Scatter(x=grouped.index, y=grouped['anomaly'], name='anomaly'))
            fig.add_trace(go.Scatter(x=grouped.index, y=grouped['label'], name='label', xaxis='x2', yaxis='y2'))
            title = 'Anomaly detected and True Label Over Time - (' + key +')'
            fig.update_layout(height=300, showlegend=True, title=title)

            # fig.show()
            fig.write_image(f"outputs/{exp_id}_{model_name}_{build}/Evaluation/ Anomaly vs Label.pdf")


            #Plot for ROC curve
            fpr, tpr, thresholds = roc_curve(grouped["label"],grouped["anomaly"])
            fig = px.area( x=fpr, y=tpr,
                title=f'ROC Curve with Scoring function : {key} <br> (AUC={auc(fpr, tpr):.4f})',
                labels=dict(x='False Positive Rate', y='True Positive Rate'),
                width=400, height=400)
            fig.add_shape(type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1)
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_xaxes(constrain='domain')
            fig.update_layout(font=dict(size=10))
            # fig.show()
            fig.write_image(f"outputs/{exp_id}_{model_name}_{build}/Evaluation/ROC Curve with Scoring function :{key}.pdf")

            #Plot for F1 Score
            target_names = ['Normal', 'Anomaly']
            report = classification_report(grouped["label"],grouped["anomaly"], target_names = target_names, output_dict=True)
            df = pd.DataFrame(report)[['Normal','Anomaly']].transpose()

            # Ploting the classification report
            fig = px.imshow(df[['precision', 'recall', 'f1-score']], 
                            color_continuous_scale='RdBu_r', text_auto=True)
            fig.update_layout(title=f'Classification Report <br> Scoring function : {key}', xaxis_title="Metrics",
                              yaxis_title="Classes",
                              xaxis=dict(side='bottom'))
            # fig.show()
            fig.write_image(f"outputs/{exp_id}_{model_name}_{build}/Evaluation/Classification Report with Scoring function :{key}.pdf")

            #Ploting confusion matrix
            confusion_matrix = metrics.confusion_matrix(grouped["label"],grouped["anomaly"])
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Normal', 'Anomaly'])
            cm_display.plot()
            # plt.show()
            plt.savefig(f'outputs/{exp_id}_{model_name}_{build}/Evaluation/Confusion Matrix with Scoring function : {key}.pdf')  

            # print('\n\nAuROC Score :',roc_auc_score(grouped["label"],grouped["scaled"], average='micro'))
            # print('\n\nconfusion matrix Score :',confusion_matrix(grouped["label"],grouped["anomaly"])) 
            # print('\n\:',classification_report(grouped["anomaly"],grouped["label"],zero_division=1))


    #Latent space visualization TSNE 


    latent = model_details['latent']
    shape = latent.shape

    latent  = np.reshape(latent, (shape[0]*shape[1], shape[2])) 
    print(latent.shape)
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(latent)
    X_embedded = pd.DataFrame(X_embedded)

    label = pd.DataFrame(model_details['label'])
    label =label.reset_index()


    label = label[:length]
    X_embedded['label'] = label[['label']].copy()
    plot = plot.reset_index()
    X_embedded['anomaly'] = plot['anomaly'].copy()
    X_embedded.columns = ['0', '1','label','anomaly']
    df1 = X_embedded








    # TSNE plot of small subset with actual label


    df = df1[2950:4050] #taking subset for visualization
    # Extract x, y, and label columns from the dataframe
    x = df['0']
    y = df['1']

    labels = df['label']

    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Loop over the unique labels
    for label in labels.unique():
        # Select the x and y values for the current label
        x_label = x[labels == label]
        y_label = y[labels == label]
        # Plot the points for the current label
        ax.scatter(x_label, y_label, label=label)

    # Add a legend to the plot
    ax.legend()

    # Show the plot
    # plt.show()
    plt.savefig(f'outputs/{exp_id}_{model_name}_{build}/Evaluation/TSNE plot with anomalies predicted.pdf')  




    ## TSNE plot of small subset with Label assigned after anomaly prediction 

    # Extract x, y, and label columns from the dataframe
    x = df['0']
    y = df['1']

    labels = df['anomaly']

    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Loop over the unique labels
    for label in labels.unique():
        # Select the x and y values for the current label
        x_label = x[labels == label]
        y_label = y[labels == label]
        # Plot the points for the current label
        ax.scatter(x_label, y_label, label=label)

    # Add a legend to the plot
    ax.legend()

    # Show the plot
    # plt.show()
    plt.savefig(f'outputs/{exp_id}_{model_name}_{build}/Evaluation/TSNE plot with True Labels given.pdf')  
    
    tcnae_plotter(model_details, exp_id, model_name,build)

    
def dghl_load_scores(scores_dir, data_dir, machines):
    labels_list = []
    scores_list = []
    for machine in machines:
        results_file = f'{scores_dir}/DGHL/{machine}/results.p'
        results = pickle.load(open(results_file,'rb'))

        scores = results['score']
        labels = pq.read_pandas(f'{data_dir}/BuildingDataset/test/{machine}.parquet').to_pandas().values
        labels = labels[:,-1]
        scores = scores.repeat(10)[-len(labels):]
        
        assert scores.shape == labels.shape, 'Wrong dimensions'

        labels_list.append(labels)
        scores_list.append(scores)
        
    return scores_list, labels_list 

def dghl_compute_f1(scores_dir, n_splits, data_dir):

    scores, labels = dghl_load_scores(scores_dir=scores_dir, data_dir=data_dir, machines=BUILDINGS)

    scores_normalized = normalize_scores(scores=scores, interval_size=96*10) # 64*1*10

    scores_normalized = np.hstack(scores_normalized)
    labels = np.hstack(labels)

    f1, precision, recall, *_ = best_f1_linspace(scores=scores_normalized, labels=labels, n_splits=n_splits, segment_adjust=True) #1000

    return f1, precision, recall

def dghl_one_liner(occlusion_intervals, occlusion_prob, data_dir):
    labels_list = []
    scores_list = []
    for machine in BUILDINGS:
        train_data = pq.read_pandas(f'{data_dir}/BuildingDataset/train/{machine}.parquet').to_pandas()
        test_data = pq.read_pandas(f'{data_dir}/BuildingDataset/test/{machine}.parquet').to_pandas()

        # Mask
        np.random.seed(1)
        train_mask = get_random_occlusion_mask(dataset=train_data[:,None,:], n_intervals=occlusion_intervals, occlusion_prob=occlusion_prob)
        train_mask = train_mask[:,0,:]
        
        train_means = np.average(train_data, axis=0, weights=train_mask)
        train_means = train_means[None, :]

        scores = np.abs(test_data - train_means)
        scores = scores.mean(axis=1)

        labels = pq.read_pandas(f'{data_dir}/BuildingDataset/test/{machine}.parquet').to_pandas().values
        labels = labels[:,-1]
        scores_list.append(scores)

    scores_normalized = normalize_scores(scores=scores_list, interval_size=96*10) # 64*1*10

    scores_normalized = np.hstack(scores_normalized)
    labels = np.hstack(labels_list)

    f1, precision, recall, *_ = best_f1_linspace(scores=scores_normalized, labels=labels, n_splits=100, segment_adjust=True)
        
    return f1, precision, recall

def dghl_nn(occlusion_intervals, occlusion_prob, data_dir):
    downsampling_size = 10
    labels_list = []
    scores_list = []
    for machine in BUILDINGS:
        train_data = pq.read_pandas(f'{data_dir}/BuildingDataset/train/{machine}.parquet').to_pandas()
        test_data = pq.read_pandas(f'{data_dir}/BuildingDataset/test/{machine}.parquet').to_pandas()
        train_data = train_data[:, None, :]
        test_data = test_data[:, None, :]
        len_test = len(test_data)
        
        # calling dghl_normalization
        train_data = dghl_normalization(train_data)
        test_data = dghl_normalization(test_data)
        # Padding
        dataset = np.vstack([train_data, test_data])
        
        # Mask
        np.random.seed(1)
        train_mask = get_random_occlusion_mask(dataset=train_data[:, None, :], n_intervals=occlusion_intervals, occlusion_prob=occlusion_prob)
        train_mask = train_mask[:, 0, :]

        train_data = train_data*train_mask

        nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(train_data)
        distances, _ = nbrs.kneighbors(test_data)
        scores = distances.mean(axis=1)        

        labels =  pq.read_pandas(f'{data_dir}/BuildingDataset/test/{machine}.parquet').to_pandas().values
        labels = labels[:,-1]
        labels = labels[len_test//2:, 0]
        scores = scores.repeat(10)[-len(labels):]

        labels_list.append(labels)
        scores_list.append(scores)

    scores_normalized = normalize_scores(scores=scores_list, interval_size=96*10)
    scores_normalized = np.hstack(scores_normalized)
    labels = np.hstack(labels_list)

    f1, precision, recall, *_ = best_f1_linspace(scores=scores_normalized, labels=labels, n_splits=100, segment_adjust=True)
        
    return f1, precision, recall


    

# from experiment import LSTMAE_Experiment
# from src.models import lstmae
# PROJECT_DIR = Path(__file__).parent
# sys.path.append(str(PROJECT_DIR))
#p = Path(PROJECT_DIR, "outputs", rf"{exp_id}_{model_name}")
# with open(rf"outputs/{exp_id}_{model_name}/exp.pkl","rb") as op:
#     exp = pickle.load(op) 
