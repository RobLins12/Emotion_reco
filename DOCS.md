# Documentação do Código de Reconhecimento de Emoções

Este documento fornece uma explicação detalhada do código de reconhecimento de emoções que utiliza OpenCV e TensorFlow. O código captura vídeo em tempo real, detecta rostos e prediz a emoção presente.

## Requisitos

Para executar este código, você precisará instalar as seguintes bibliotecas:

- OpenCV (cv2)
- NumPy (numpy)
- TensorFlow (tensorflow)

## Estrutura do Código

O código é dividido nas seguintes seções:

1. [Importações e Carregamento do Modelo](#1-importações-e-carregamento-do-modelo)
2. [Funções de Detecção e Processamento de Imagens](#2-funções-de-detecção-e-processamento-de-imagens)
3. [Função de Predição de Emoções](#3-função-de-predição-de-emoções)
4. [Captura de Vídeo e Loop Principal](#4-captura-de-vídeo-e-loop-principal)

### 1. Importações e Carregamento do Modelo

```python
import cv2 as cv
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

loaded_model = load_model("model_50%.h5")  # Insira aqui o path para o seu modelo
```

O modelo de reconhecimento de emoções é carregado a partir de um arquivo .h5:

### 2. Funções de Detecção e Processamento de Imagens

A função detect_bounding_box detecta rostos em um frame de vídeo usando um classificador Haar Cascade:

```python
face_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_bounding_box(vid):
    gray_image = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces
```
A função load_and_preprocess_frame redimensiona e normaliza um frame de vídeo para preparação para a predição:

```python
def load_and_preprocess_frame(frame):
    resized_frame = cv.resize(frame, (150, 150))
    img_array = np.array(resized_frame, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, resized_frame
```

### 3. Função de Predição de Emoções

A função emotion_prediction utiliza o modelo carregado para predizer a emoção presente em um frame de vídeo:

```python
def emotion_prediction(frame, model):
    img_array, _ = load_and_preprocess_frame(frame)
    prediction = model.predict(img_array)
    position = np.argmax(prediction)
    emotion = ""
    match position:
        case 0:
            emotion = "Raiva"
        case 1:
            emotion = "Nojo"
        case 2:
            emotion = "Medo"
        case 3:
            emotion = "Feliz"
        case 4:
            emotion = "Neutro"
        case 5:
            emotion = "Triste"
        case 6:
            emotion = "Surpresa"
    return prediction, emotion
```

### 4. Captura de Vídeo e Loop Principal

A captura de vídeo é iniciada e processada em um loop contínuo. As emoções detectadas são exibidas no vídeo em tempo real:

```python
video_capture = cv.VideoCapture(0)

while True:

    result, video_frame = video_capture.read()  # lê frames do vídeo

    if result is False:
        break 

    _, emotion = emotion_prediction(frame=video_frame, model=loaded_model)

    font = cv.FONT_HERSHEY_SIMPLEX

    faces = detect_bounding_box(video_frame)  

    cv.putText(video_frame, emotion, (50, 50), font, 1, (0, 255, 255), 2, cv.LINE_AA)

    cv.imshow("Emotion Reco", video_frame)  

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv.destroyAllWindows()
```

### Como Executar o Código

1. Certifique-se de que todas as dependências estão instaladas.
2. Altere o caminho do modelo na linha loaded_model = load_model("model_50%.h5") para o local correto do seu modelo.
3. Execute o script.

### Conclusão

Este código captura vídeo em tempo real, detecta rostos e prediz as emoções usando um modelo pré-treinado. É uma aplicação prática de visão computacional e aprendizado de máquina para reconhecimento de emoções.