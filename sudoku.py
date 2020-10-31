import cv2
import numpy as np
import time
import keras

from keras.models import load_model


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect



def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped



#xac dinh khung sudoku
def read_sudoku(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 127, 255, 0)
	tmp = cv2.findContours( thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = tmp[0] if len(tmp) == 2 else tmp[1]
	cnts = sorted(contours, key=lambda ctr: cv2.contourArea(ctr), reverse=True)
	if len(cnts) > 1:
		cnt = cnts[1]
		epsilon = 0.1*cv2.arcLength(cnt,True)
		approx = []
		approx = cv2.approxPolyDP(cnt,epsilon,True)
		if approx is not None and len(approx) >= 3:
			pts = np.asarray([approx[i][0] for i in range(len(approx))])
		else: 
			pts = np.array([[0,0], [0,0], [0,0], [0,0]])
		cv2.drawContours(frame, approx, -1, (0,0,255), 5)
		print(approx)
		# sudoku = frame[approx[0][0][1]:approx[2][0][1], approx[0][0][0]:approx[2][0][0]]
		

		print(pts)
		warped = four_point_transform(frame, pts)
		warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
		ret,sudoku = cv2.threshold(warped_gray,127,255,cv2.THRESH_BINARY)
		sudoku =  cv2.bitwise_not(sudoku)

		matrix = list()

		if sudoku.shape[1] > 500:
			w = round(sudoku.shape[1]/9)
			h = round(sudoku.shape[2]/9)
			for i in range(10):
				for j in range(10):
					im = sudoku[i*w:(i+1)*w, j*h : (j+1)*h]
					if im.shape[1] > 55:
						im = cv2.resize(im,(28,28))
						im2arr = np.array(im)
						im2arr = im2arr.reshape(1,28,28,1)
						number_predict = model.predict_classes(im2arr)
						matrix.append(number_predict)
						
			print(matrix)	
		cv2.imshow("result",warped)
		#Add note on result image
				
	cv2.imshow("gray", gray)
	cv2.imshow("sudoku", sudoku)	

	# cv2.imshow("Sudoku", sudoku)
# ret,thresh_img = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# 		x = np.invert(thresh_img)
# 		x[approx[0][0][1]:approx[2][0][1], approx[0][0][0]:approx[2][0][0]] = 255


# image = cv2.imread("sudoku_web1.jpg")
# image = cv2.resize(image,(500,600))

# read_sudoku(image)
model = load_model('model.h5')

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
cap = cv2.VideoCapture(0)
while(True):
	time.sleep(0.75)
	ret, frame = cap.read()
	read_sudoku(frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
