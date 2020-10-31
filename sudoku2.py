import cv2
import numpy as np
import matplotlib.pyplot as plt



im = cv2.imread('sudoku.png')
font = cv2.FONT_ITALIC

w = im.shape[1]/9
h = im.shape[0]/9

# row1 = im[0:y,:]
# im = cv2.resize(im,(500, 600))
# im = cv2.putText(im,'2',(40,60), font, 1,(255,0,0),2,cv2.LINE_AA)

print (w,h)

# cv2.imshow("sdfsf",im[0:w,h*2:h*3])

# cv2.imshow("im", im)
cv2.waitKey(0)
cv2.destroyAllWindows()


