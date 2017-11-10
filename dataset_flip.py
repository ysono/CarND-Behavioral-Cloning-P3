import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1])
print(img.shape)

flipped = np.fliplr(img)
cv2.imwrite(sys.argv[2], flipped)
