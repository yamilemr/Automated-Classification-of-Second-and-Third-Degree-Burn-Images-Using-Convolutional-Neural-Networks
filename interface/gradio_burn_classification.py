import cv2
import numpy as np
import gradio as gr
from keras import models

#Required input dimensions for the proposed CNN architecture
WIDTH = 540
HEIGHT = 960

#Load the model
cnn = models.load_model('models/proposed_model/burn_green_cnn.keras')

def predict_burn_degree(img):
    '''
    Function that receives an image from the Gradio interface, preprocesses it 
    using the same pipeline applied during model training, and returns the
    predicted burn degree along with its associated probability.

    Parameters:
    - img (numpy.ndarray): Input image in RGB format.

    Returns:
    - degree (int): Predicted burn degree (2 or 3).
    - probability (float): Associated probability rounded to 4 decimals.
    '''
    #The image is received as a numpy array in RGB format; validate it is not null
    if img is None:
        return None, {'error': 'Invalid image'}
    
    #Ensure the image has 3 channels (RGB)
    if img.ndim == 2:
        #If the image is grayscale, convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        #If the image has an alpha channel (RGBA), convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    #Preprocessing 
    resized = cv2.resize(img, (WIDTH, HEIGHT)) #Resize the image
    green = resized[:, :, 1] #Extract only the green channel
    X = np.expand_dims(green, axis=(0, -1)) #Reshape the array to be compatible with Keras: (1, H, W, 1)

    #Prediction 
    pred = cnn.predict(X)
    prob = pred[0][0] #Probability of belonging to class 1 (third degree burn)
    
    #If probability > 0.5, it's a third degree burn. If probability <= 0.5, it's a second-degree burn
    if prob > 0.5:
        degree = 3
        probability  = float(prob)
    else:
        degree = 2
        probability  = float(1 - prob)

    return degree, round(probability , 4)

#Gradio interface definition
iface = gr.Interface(
    fn=predict_burn_degree,
    inputs=gr.Image(type='numpy', label='Upload an image'),
    outputs=[
        gr.Number(label='Burn degree'),
        gr.Number(label='Probability')],
    title='Automated Classification of Second and Third Degree Burn Images',
    description='Upload an image of a burn and the model will predict its degree. Try to use a clear image where the burn covers the entire frame.',
    allow_flagging='never'
)

#Launch the application
iface.launch(share=False) #False runs locally only; True generates a temporary public link