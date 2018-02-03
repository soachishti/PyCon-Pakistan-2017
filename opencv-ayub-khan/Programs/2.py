import cv2
from matplotlib import pyplot as plt


# Playing around with different dimensions
image = cv2.imread('bgr.png')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

#Blue
image_blue = image[:, :, 0]
plt.imshow(image_blue)
plt.show()

#Green
image_green = image[:, :, 1]
plt.imshow(image_green)
plt.show()

#Red
image_red = image[:, :, 2]
plt.imshow(image_red)
plt.show()


