import cv2
cam = cv2.VideoCapture(0)

while True:
  ret, image = cam.read()
  cv2.imshow('ImageTest', image)
  k = cv2.waitKey(1)
  break
cv2.imwrite('/home/pi/testimage.jpg', image)
cam.release()
cv2.destroyAllWindows()