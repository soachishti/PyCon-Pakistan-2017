import cv2
from matplotlib import pyplot as plt


# Simple Image Editing

image = cv2.imread('afridi.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

image[330:370, 140:180] = image[372:412, 7:47]
image[372:412, 7:47] = image[412:452, 7:47]

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

