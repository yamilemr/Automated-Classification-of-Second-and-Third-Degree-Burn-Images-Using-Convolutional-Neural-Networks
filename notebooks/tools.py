import cv2
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def load_images_with_labels(path=None, channels=None, width=540, height=960):
    '''
    Function that loads images from a directory structured by classes (second_degree and third_degree),
    resizes them, selects the desired color channels, and generates their corresponding labels.

    Parameters:
    - path (str): Path to the root directory that contains the folders with images for each class.
    - channels (str or list): Defines which color channels to use. It can be 'bgr' to use the full image
                              or a combination of ['blue', 'green', 'red'].
    - width (int): Target width to resize images.
    - height (int): Target height to resize images.

    Returns:
    - X (np.ndarray): Array of processed images.
    - y (np.ndarray): Array of labels associated with each image.
    '''
    #If no channels are specified, use the three BGR channels
    if channels == None:
        channels = 'bgr'

    #Input path validation
    if path is None:
        raise ValueError("The 'path' parameter cannot be None")

    #Lists to store images and labels
    X = []
    y = []

    #Iterate over each folder (class) inside the root directory
    for folder in os.listdir(path):
        #Iterate over each image inside the class folder
        for image in os.listdir(path + folder):
            img = cv2.imread(path + folder + '/' + image) #Read the image
            resized_img = cv2.resize(img, (width, height)) #Resize the image
            
            #If all BGR channels are used, add the full image
            if channels == 'bgr':
                X.append(resized_img)
            
            #Extract the selected channels from the image
            else:
                selected_channels = [] #List to store selected channels

                if 'blue' in channels:
                    blue = resized_img[:, :, 0]
                    selected_channels.append(blue)

                if 'green' in channels:
                    green = resized_img[:, :, 1]
                    selected_channels.append(green)

                if 'red' in channels:
                    red = resized_img[:, :, 2]
                    selected_channels.append(red)
                
                #If there is only one channel, add it directly
                if len(selected_channels) == 1:
                    X.append(selected_channels[0])
                
                #If there is more than one channel, stack them into a single array
                else:
                    X.append(np.dstack(selected_channels))

            #0: second degree (negative class), 1: third degree (positive class)
            if folder == 'second_degree':
                y.append(0) 
            else:
                y.append(1)

    #Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    #If there is only one channel, expand dimensions to be compatible with the model
    if X.ndim == 3:
        X = X[..., np.newaxis]

    return X, y


def calculate_metrics(model, X, y):
    '''
    Function that computes performance metrics for a binary classification model.

    Parameters:
    - model (keras.Model): Trained keras model with a 'predict' method.
    - X (np.ndarray): Input dataset.
    - y (np.ndarray): Labels associated with the input data.

    Returns:
    - CM (np.ndarray): Confusion matrix.
    - accuracy (float): Model accuracy.
    - precision (float): Model precision.
    - recall (float): Model recall.
    - f1 (float): Model F1-score.
    '''
    predictions = model.predict(X) #Model predictions

    y_true = y.flatten() #Flatten labels to ensure compatibility
    y_pred = (predictions > 0.5).astype(int).flatten() #Convert probabilities to binary classes. Probability > 0.5 corresponds to third degree (class 1)
    
    #Compute confusion matrix and evaluation metrics
    CM = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return CM, accuracy, precision, recall, f1


def evaluate_model(model, X, y, dataset='test'):
    '''
    Function that evaluates the performance of a binary classification model, visualizes the confusion matrix using a 
    heatmap, and includes the performance metrics in the figure.

    Parameters:
    - model (keras.Model): Trained keras model with a 'predict' method.
    - X (np.ndarray): Input dataset.
    - y (np.ndarray): Labels associated with the input data.
    - dataset (str): Name of the evaluated dataset (e.g., 'training', 'validation', or 'test').

    Returns:
    - None. The function generates a confusion matrix plot and includes the evaluation metrics.
    '''
    CM, accuracy, precision, recall, f1 = calculate_metrics(model, X, y) #Compute confusion matrix and evaluation metrics

    #Define a custom color gradient for heatmap visualization
    colors = ['#e7eeff', '#8882d9', '#264f73']
    custom_cmap = LinearSegmentedColormap.from_list("custom_bupu", colors)

    plt.figure(figsize=(4, 4)) #Initialize figure

    #Plot confusion matrix as heatmap
    sns.heatmap(CM, annot=True, fmt='d', cmap=custom_cmap,
                xticklabels=['2nd degree', '3rd degree'],
                yticklabels=['2nd degree', '3rd degree'],
                annot_kws={'size': 13})
    
    #Set title and axis labels
    plt.title(dataset, fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)

    #Display metrics below the plot
    plt.figtext(0.45, 0, f'\nAccuracy: {accuracy:.4f} | Precision: {precision:.4f}\nRecall: {recall:.4f} | F1-Score: {f1:.4f}',
                ha='center', fontsize=11, va='top')
    

def save_metrics_to_csv(model, X, y, model_name='OurModel', dataset='test', csv_path='metrics.csv'):
    '''
    Function that evaluates the performance of a binary classification model and saves its metrics to a CSV file.
    The metrics are stored according to the evaluated dataset (training, validation, or test) and are organized by model.

    Parameters:
    - model (keras.Model): Trained keras model with a 'predict' method.
    - X (np.ndarray): Input dataset.
    - y (np.ndarray): Labels associated with the input data.
    - model_name (str): Name of the model to be registered in the CSV.
    - dataset (str): Evaluated dataset ('train', 'val', or 'test').
    - csv_path (str): Path to the CSV where the metrics are stored.

    Returns:
    - None. The function saves or updates the model metrics in a CSV file.
    '''
    _, accuracy, precision, recall, f1 = calculate_metrics(model, X, y) #Compute the model evaluation metrics

    #Create CSV if it does not exist
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=['Model', 
                                   'Accuracy Training', 'Precision Training', 'Recall Training', 'F1 Training', 
                                   'Accuracy Validation', 'Precision Validation', 'Recall Validation', 'F1 Validation', 
                                   'Accuracy Test', 'Precision Test', 'Recall Test', 'F1 Test'])
        df.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path) #Load the CSV file

    #Create model row if it does not exist
    if model_name not in df['Model'].values:
        df = pd.concat([df, pd.DataFrame({'Model': [model_name]})], ignore_index=True)

    #Dictionary that maps each dataset to its corresponding columns
    columns = {'train': ('Accuracy Training', 'Precision Training', 'Recall Training', 'F1 Training'),
               'val':   ('Accuracy Validation', 'Precision Validation', 'Recall Validation', 'F1 Validation'),
               'test':  ('Accuracy Test', 'Precision Test', 'Recall Test', 'F1 Test')}

    df_accuracy, df_precision, df_recall, df_f1 = columns[dataset] #Select the columns according to the evaluated dataset

    #Update the model metrics in the CSV
    df.loc[df['Model'] == model_name, df_accuracy] = accuracy
    df.loc[df['Model'] == model_name, df_precision] = precision
    df.loc[df['Model'] == model_name, df_recall] = recall
    df.loc[df['Model'] == model_name, df_f1] = f1

    df.to_csv(csv_path, index=False) #Save changes to the CSV