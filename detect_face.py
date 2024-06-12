import cv2 as cv

face_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_bounding_box(vid):
    gray_image = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

video_capture = cv.VideoCapture(0)

while True:

    result, video_frame = video_capture.read()  # read frames from the video

    if result is False:
        break 

    faces = detect_bounding_box(video_frame)  

    cv.imshow("Face Detection", video_frame)  

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv.destroyAllWindows()