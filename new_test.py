import cv2
import keras
from keras.models import load_model
import numpy as np 


model = load_model('model.h5')

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])



img = cv2.imread('black.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh_img = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
img_rows, img_cols = 28,28
thresh_img = cv2.bitwise_not(thresh_img)
img = cv2.resize(thresh_img,(28,28))
im2arr = np.array(img)
im2arr = im2arr.reshape(1,28,28,1)
classes = model.predict_classes(im2arr)
print(classes)
