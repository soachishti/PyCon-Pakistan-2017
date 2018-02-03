import cv2
import pyscreenshot as ScreenShot
import numpy as np


cv2.namedWindow('vid', cv2.WINDOW_KEEPRATIO)

while True:
    img = np.array(ScreenShot.grab().convert('RGB'))
    cv2.imshow('vid',img)
    key = cv2.waitKey(1)
    if key==27:
        break
cv2.destroyAllWindows()
