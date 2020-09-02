import numpy as np
import cv2
import pickle
from playsound import playsound
import sqlite3
from datetime import datetime
t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

# recognizer=cv2.face.EigenFaceRecognizer_create()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)
writeknown = True


def my_function():
    writeknown = False


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        # print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]  # (ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]

        # recognize? deep learned model predict keras tensorflow pytorch scikit learn
        id_, conf = recognizer.predict(roi_gray)

        if conf >= 4 and conf <= 85:
            
            # print(5: #id_)
            # print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1,
                        color, stroke, cv2.LINE_AA)
            my_function()
        else:
            playsound('1.mp3')

        img_item = "7.png"
        cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0)  # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        # subitems = smile_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in subitems:
        #	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # Display the resulting frame

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
if writeknown is True:
    db = sqlite3.connect('db.db')

    qry = "insert into KNOWN (PersonName, time) values(?,?);"
    info = [(name, t)]
    cur = db.cursor()
    cur.executemany(qry, info)
    db.commit()
    print("one record added successfully to kown")
    db.rollback()
    db.close()
else:
    db = sqlite3.connect('db.db')
    qry = "insert into UNKNOWN (time) values(?);"
    info = [(t)]
    cur = db.cursor()
    cur.executemany(qry, info)
    db.commit()
    print("one record added successfully to unkown")
    db.rollback()
    db.close()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
