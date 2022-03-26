import cv2
import os
import numpy as np
from keras.models import load_model
from datetime import datetime


model = load_model("my_model.h5")

labels_dict = {0: 'without mask', 1: 'mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

size = 4
for i in os.listdir('./train/with_mask'):
    im=cv2.imread('./train/with_mask/'+i)
    if(im is None):
        continue
    classifier = cv2.CascadeClassifier('./venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    im = cv2.flip(im, 1, 1)
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))


    faces = classifier.detectMultiScale(mini)
    for f in faces:
            (x, y, w, h) = [v * size for v in f]
            face_img = im[y:y + h, x:x + w]
            resized = cv2.resize(face_img, (150, 150))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 150, 150, 3))
            reshaped = np.vstack([reshaped])
            result = model.predict(reshaped)
            # print(result)

            label = np.argmax(result, axis=1)[0]

            cv2.rectangle(im, (x, y), (x + w, y + h), color_dict[label], 2)
            cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1)
            cv2.putText(im, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            dt = str(datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            frame = cv2.putText(im, dt,
                                (10, 20) ,
                                font, 0.5,
                                (255, 255, 255),
                                1, cv2.LINE_8)
            dt = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            cv2.imwrite('./withoutmask/'+i, im)

cv2.destroyAllWindows()