import cv2
from matplotlib import pyplot as plt
import numpy as np


# The image we are going to use for system training.
digits_learning_image = cv2.imread('hand_written_data/digits.png')
cv2.imshow('image', digits_learning_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#NEED TO RECHECK
gray_digits_learning_image = cv2.cvtColor(digits_learning_image, cv2.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size
digit_cells = [np.hsplit(row, 100) for row in np.vsplit(gray_digits_learning_image, 50)]
print 'Digit cells length : ', len(digit_cells)
print 'Digit cells width : ', len(digit_cells[1])

# Make it into a Numpy array. It's size will be (50,100,20,20)
digits_list = np.array(digit_cells)

# Now we prepare train_data and test_data.
digits_for_training = digits_list[:, :50].reshape(-1,400).astype(np.float32)  # Size=(2500,400)
digits_for_testing = digits_list[:, 50:100].reshape(-1,400).astype(np.float32)# Size=(2500,400)

# Create labels for training data and testing data
numbers = np.arange(10)
labels_for_training_data = np.repeat(numbers, 250)[:, np.newaxis]
labels_for_testing_data = labels_for_training_data.copy()

print "Shape of training data : ", digits_for_training.shape
print "Shape of lables for training data : ", labels_for_training_data.shape

# Initiate kNN, train the data, then test it with test data for k=4
knn = cv2.ml.KNearest_create()
knn.train(digits_for_training, cv2.ml.ROW_SAMPLE, labels_for_training_data)
ret, result, neighbours, dist = knn.findNearest(digits_for_testing, k=4)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result == labels_for_testing_data
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size

print '************************* DIGITS OCR *************************'
print 'matches : ', matches
print 'correct : ', correct
print 'accuracy : ', accuracy

# Test Data
digit_image_for_testing = cv2.imread('hand_written_data/1.png')
# digit_image_for_testing = cv2.imread('hand_written_data/2.png')
# digit_image_for_testing = cv2.imread('hand_written_data/3.png')
# digit_image_for_testing = cv2.imread('hand_written_data/4.png')
# digit_image_for_testing = cv2.imread('hand_written_data/5.png')
# digit_image_for_testing = cv2.imread('hand_written_data/6.png')
# digit_image_for_testing = cv2.imread('hand_written_data/7.png')
# digit_image_for_testing = cv2.imread('hand_written_data/8.png')
# digit_image_for_testing = cv2.imread('hand_written_data/9.png')


plt.imshow(digit_image_for_testing)
plt.show()

gray_image = cv2.cvtColor(digit_image_for_testing, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image)
plt.show()

ret, thresh = cv2.threshold(gray_image, 10, 255, cv2.THRESH_OTSU)
threshold_image = cv2.threshold(thresh, 10, 255, cv2.THRESH_BINARY_INV)[1]
plt.imshow(threshold_image)
plt.show()

# Perform the resizing of the image according to our train data (20, 20)
r = 20.0 / threshold_image.shape[1]
dim = (20, int(threshold_image.shape[0] * r))
resized_image = cv2.resize(threshold_image, dim, interpolation=cv2.INTER_AREA)
plt.imshow(resized_image)
plt.show()

# Convert to linear image and recognize the digit using KNN.
digit_in_1d_array = resized_image.reshape(-1,400).astype(np.float32)
ret, result, neighbours, dist = knn.findNearest(digit_in_1d_array, k=4)

print '************************* DIGITS OCR *************************'
print 'ret : ', ret
print 'result : ', result
print 'neighbours : ', neighbours
print 'dist : ', dist
