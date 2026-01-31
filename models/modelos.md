## mobileNetV2_rgb.keras
- Este modelo está basado en la arquitectura MobileNetV2 y utiliza transfer learning con los pesos de ImageNet
- Se normalizaron las imágenes con preprocess_input de MobileNetV2
- Se entrenó con las imágenes en RGB
- Se congelaron las capas base
- La cabeza de clasificación se remplazó por una capa GlobalAveragePooling2D, seguida de una capa Dropout con una tasa de 0.5, y una capa densa con una neurona y función de activación sigmoide
- Se usó EarlyStopping y ReduceLROnPlateau
- El entrenamiento se detuvo en 74 épocas
- Learning rate mínimo: 6.2500e-05
- Accuracy train: 0.9778
- Accuracy validation: 0.9111
- Accuracy test: 0.7500
- El history de entrenamiento es `history_mobileNetV2_rgb.pkl`

## vgg16_rgb.keras
- Este modelo está basado en la arquitectura VGG16 y utiliza transfer learning con los pesos de ImageNet
- Se normalizaron las imágenes con preprocess_input de VGG16
- Se entrenó con las imágenes en RGB
- Se congelaron las capas base
- La cabeza de clasificación se remplazó por una capa GlobalAveragePooling2D, seguida de una capa Dropout con una tasa de 0.5, y una capa densa con una neurona y función de activación sigmoide
- Se usó EarlyStopping y ReduceLROnPlateau
- El entrenamiento se detuvo en 64 épocas
- Learning rate mínimo: 1.2500e-04
- Accuracy train: 0.9556
- Accuracy validation: 0.9333
- Accuracy test: 0.7500
- El history de entrenamiento es `history_vgg16_rgb.pkl`

## resNet50_rgb.keras
- Este modelo está basado en la arquitectura ResNet50 y utiliza transfer learning con los pesos de ImageNet
- Se normalizaron las imágenes con preprocess_input de ResNet50
- Se entrenó con las imágenes en RGB
- Se congelaron las capas base
- La cabeza de clasificación se remplazó por una capa GlobalAveragePooling2D, seguida de una capa Dropout con una tasa de 0.5, y una capa densa con una neurona y función de activación sigmoide
- Se usó EarlyStopping y ReduceLROnPlateau
- El entrenamiento se detuvo en 45 épocas
- Learning rate mínimo: 5.0000e-04
- Accuracy train: 0.9833
- Accuracy validation: 0.9556
- Accuracy test: 0.7708
- El history de entrenamiento es `history_resNet50_rgb.pkl`

## ourModel_green.keras
- Mejor modelo de la búsqueda secuencial, pero sólo con el canal verde
- Se usó EarlyStopping y ReduceLROnPlateau
- El entrenamiento se detuvo en 85 épocas
- Learning rate mínimo: 1.5625e-05
- Accuracy train: 0.9889
- Accuracy validation: 0.9778
- Accuracy test: 0.9375
- El history de entrenamiento es `history_ourModel_green.pkl`