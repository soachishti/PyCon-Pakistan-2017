import cv2
from matplotlib import pyplot as plt


# Read Image
image = cv2.imread('hello-world.jpg')
print 'Image shape (hight, width, dimensions) : %s ' % str(image.shape)
print image.size
print image.dtype

# Show Image
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# To Show in JupyterNoteBood
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
