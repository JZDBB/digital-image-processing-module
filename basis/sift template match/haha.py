import matplotlib.pyplot as plt
import cv2
imgname2 = "template_last.jpg"

img2 = cv2.imread(imgname2)

plt.imshow(img2, cmap='gray')
plt.show()

img = img2[852:896, 434:600]
cv2.imwrite('template_1.jpg', img)