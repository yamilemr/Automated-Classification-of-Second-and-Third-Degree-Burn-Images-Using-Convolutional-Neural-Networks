import cv2
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def load_images_with_labels(path=None, channels=None):
    width = 540
    height = 960
    
    if channels == None:
        channels = 'bgr'

    if path is None:
        raise ValueError("The 'path' parameter cannot be None")
    folder_path  = path
        
    X = []
    y = []

    for i in os.listdir(folder_path ):
        for j in os.listdir(folder_path  + i):
            img = cv2.imread(folder_path  + i + '/' + j)
            resized_img = cv2.resize(img, (width, height))
            
            if channels == 'bgr':
                X.append(resized_img)
            
            else:
                selected_channels = []

                if 'blue' in channels:
                    blue = resized_img[:, :, 0]
                    selected_channels.append(blue)

                if 'green' in channels:
                    green = resized_img[:, :, 1]
                    selected_channels.append(green)

                if 'red' in channels:
                    red = resized_img[:, :, 2]
                    selected_channels.append(red)
                
                if len(selected_channels) == 1:
                    X.append(selected_channels[0])
                else:
                    X.append(np.dstack(selected_channels))

            #0: second degree (negative class), 1: third degree (positive class)
            if i == 'second_degree':
                y.append(0) 
            else:
                y.append(1)

    X = np.array(X)
    y = np.array(y)

    #If there is only one channel, expand dimensions to be compatible with the model
    if X.ndim == 3:
        X = X[..., np.newaxis]

    return X, y


def calculate_metrics(model, X, y):
    predictions = model.predict(X)
    y_true = y.flatten()
    y_pred = (predictions > 0.5).astype(int).flatten() #Probability greater than 0.5 corresponds to third grade (class 1)
    
    CM = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return CM, accuracy, precision, recall, f1


def evaluate_model(model, X, y, dataset='test'):
    CM, accuracy, precision, recall, f1 = calculate_metrics(model, X, y)

    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#e7eeff', '#8882d9', '#264f73']
    custom_cmap = LinearSegmentedColormap.from_list("custom_bupu", colors)

    plt.figure(figsize=(4, 4))
    sns.heatmap(CM, annot=True, fmt='d', cmap=custom_cmap,
                xticklabels=['2nd degree', '3rd degree'],
                yticklabels=['2nd degree', '3rd degree'],
                annot_kws={'size': 13})
    plt.title(dataset, fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.figtext(0.45, 0, f'\nAccuracy: {accuracy:.4f} | Precision: {precision:.4f}\nRecall: {recall:.4f} | F1-Score: {f1:.4f}',
                ha='center', fontsize=11, va='top')
    

def save_metrics_to_csv(model, X, y, model_name='OurModel', dataset='test', csv_path='metricas.csv'):
    _, accuracy, precision, recall, f1 = calculate_metrics(model, X, y)

    #Create CSV if it does not exist
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=['Model', 
                                   'Accuracy Train', 'Precision Train', 'Recall Train', 'F1 Train', 
                                   'Accuracy Validation', 'Precision Validation', 'Recall Validation', 'F1 Validation', 
                                   'Accuracy Test', 'Precision Test', 'Recall Test', 'F1 Test'])
        df.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)

    #Create model row if it does not exist
    if model_name not in df['Model'].values:
        df = pd.concat([df, pd.DataFrame({'Model': [model_name]})], ignore_index=True)

    columnas = {'train': ('Accuracy Train', 'Precision Train', 'Recall Train', 'F1 Train'),
                'val':   ('Accuracy Validation', 'Precision Validation', 'Recall Validation', 'F1 Validation'),
                'test':  ('Accuracy Test', 'Precision Test', 'Recall Test', 'F1 Test')}

    df_accuracy, df_precision, df_recall, df_f1 = columnas[dataset]

    df.loc[df['Model'] == model_name, df_accuracy] = accuracy
    df.loc[df['Model'] == model_name, df_precision] = precision
    df.loc[df['Model'] == model_name, df_recall] = recall
    df.loc[df['Model'] == model_name, df_f1] = f1

    df.to_csv(csv_path, index=False)