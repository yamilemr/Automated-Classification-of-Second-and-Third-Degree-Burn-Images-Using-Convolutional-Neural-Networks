import cv2
import numpy as np
import gradio as gr
from keras import models

width = 540
height = 960

# Cargar modelo
cnn = models.load_model('models/proposed_model/burn_green_cnn.keras')

def predecir_grado(img):
    # img llega como RGB (numpy) desde Gradio
    if img is None:
        return None, {"error": "Imagen no válida"}
    
    # Asegurar 3 canales RGB
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Procesamiento
    resized = cv2.resize(img, (width, height)) # Redimensionar la imagen
    green = resized[:, :, 1] # Extraer sólo el canal verde de la imagen
    X = np.expand_dims(green, axis=(0, -1)) # (1, H, W, 1)

    # Predicción
    pred = cnn.predict(X)
    prob = pred[0][0] # Probabilidad de pertenecer a la clase 1 (tercer grado)
    if prob > 0.5:
        grado = 3
        probabilidad = float(prob)
    else:
        grado = 2
        probabilidad = float(1 - prob)

    return grado, round(probabilidad, 4)

iface = gr.Interface(
    fn=predecir_grado,
    inputs=gr.Image(type="numpy", label="Sube una imagen"),
    outputs=[
        gr.Number(label="Grado de la quemadura"),
        gr.Number(label="Probabilidad")
    ],
    title="Clasificación de Quemaduras de 2° y 3° Grado",
    description="Sube una imagen de una quemadura y el modelo predecirá el grado. Trata que la imagen sea clara y que la quemadura cubra toda la imagen.",
    allow_flagging="never"
)

iface.launch(share = False) # True para compartir en línea, False se queda en local