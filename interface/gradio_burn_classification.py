import cv2
import numpy as np
import gradio as gr
from keras import models

# Dictionary with model names and their respective paths
MODEL_PATHS = {
    'Proposed Model': 'models/proposed_model/burn_green_cnn.keras',
    'MobileNetV2': 'models/mobilenetv2/mobilenetv2.keras',
    'ResNet50': 'models/resnet50/resnet50.keras',
    'VGG16': 'models/vgg16/vgg16.keras' 
}

# Dictionary with input dimensions (Width, Height) required by each model.
# Adjust the 224x224 values if you used a different resolution during training.
MODEL_DIMS = {
    'Proposed Model': (540, 960),
    'MobileNetV2': (224, 224),
    'ResNet50': (224, 224),
    'VGG16': (224, 224)
}

# Global variables to avoid loading the model on every prediction (saves RAM and time)
current_model_name = None
cnn = None

def predict_burn_degree(img, model_choice):
    
    '''
    Function that receives an image and the selected model from the Gradio interface, preprocesses it 
    using the same pipeline applied during model training, and returns the
    predicted burn degree along with its associated probability.

    Parameters:
    - img (numpy.ndarray): Input image in RGB format.
    - model_choice (string): Selected model name

    Returns:
    - degree (int): Predicted burn degree (2 or 3).
    - probability (float): Associated probability rounded to 4 decimals.
    '''
    global current_model_name, cnn
    
    # Load the model dynamically only if it's different from the one already in memory
    if current_model_name != model_choice:
        print(f"Cargando el modelo: {model_choice}...")
        cnn = models.load_model(MODEL_PATHS[model_choice])
        current_model_name = model_choice

    # Validate that the image is not None
    if img is None:
        return None, {'error': 'Invalid image'}
    
    # Ensure the image has 3 channels (RGB)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Dynamic preprocessing for each model
    target_width, target_height = MODEL_DIMS[model_choice]
    resized = cv2.resize(img, (target_width, target_height))
    
    # Separate logic: green channel vs full RGB
    if model_choice == 'Proposed Model':
        input_channel = resized[:, :, 1] # Extract only the green channel
        # Reshape for Keras: (Height, Width) -> (1, Height, Width, 1)
        X = np.expand_dims(input_channel, axis=(0, -1)) 
    else:
        # Reshape for Keras: (Height, Width, 3) -> (1, Height, Width, 3)
        X = np.expand_dims(resized, axis=0) 

    # Inference
    pred = cnn.predict(X)
    prob = pred[0][0] # probability to being a third degree burn (class 1)
    
    # Classification logic
    if prob > 0.5:
        degree = 3
        probability = float(prob)
    else:
        degree = 2
        probability = float(1 - prob)

    return degree, round(probability, 4)

# Initialize Gradio interface
iface = gr.Interface(
    fn=predict_burn_degree,
    inputs=[
        gr.Image(type='numpy', label='Upload an image', height=360),
        gr.Dropdown(
            choices=['Proposed Model', 'MobileNetV2', 'ResNet50', 'VGG16'], 
            value='Proposed Model', # default value
            label='Select Model'
        )
    ],
    outputs=[
        gr.Number(label='Burn degree'),
        gr.Number(label='Probability')
    ],
    title='Automated Classification of Second- and Third-Degree Burn Images',
    description='Upload an image of a burn, select a classification model, and get the predicted degree. Try to use a clear image where the burn covers the entire frame.',
    allow_flagging='never'
)

# Launch app
iface.launch(share=False)