import cv2
from matplotlib import pyplot as plt


#Image Crope and Save

#Load a new image
image = cv2.imread('cute-kid.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

#Crope
image = image[0:500, 300:650]

# Write 
cv2.imwrite('cropped_image.png', image)

# Load the Cropped image and show
cropped_image = cv2.imread('cropped_image.png')
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.show()
