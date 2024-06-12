import cv2 as cv

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import load_model # type: ignore

loaded_model = load_model("model.h5")

face_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_bounding_box(vid):
    gray_image = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

def load_and_preprocess_frame(frame):
    resized_frame = cv.resize(frame, (150, 150))
    img_array = np.array(resized_frame, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, resized_frame

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

video_capture = cv.VideoCapture(0)

while True:

    result, video_frame = video_capture.read()  # read frames from the video

    if result is False:
        break 

    _ , emotion=emotion_prediction(frame=video_frame, model=loaded_model)

    font = cv.FONT_HERSHEY_SIMPLEX

    faces = detect_bounding_box(video_frame)  

    cv.putText(video_frame, emotion, (50, 50), font, 1, (0, 255, 255), 2, cv.LINE_AA)

    cv.imshow("Face Detection", video_frame)  

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv.destroyAllWindows()