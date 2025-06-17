
import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
IMAGE_PATH = '1.jpg'
image = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(bfilter, 30, 200)
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = keypoints[0] if len(keypoints) == 2 else keypoints[1]
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
result = image.copy()
if location is not None:
    cv2.drawContours(result, [location], 0, (0,255,0), 3)
else:
    print("❌ License plate not found.")
    exit()
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(image, image, mask=mask)
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
cropped = gray[topx:bottomx+1, topy:bottomy+1]
reader = easyocr.Reader(['fa'])
result_text = reader.readtext(cropped)
plt.figure(figsize=(10,7))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Detected License Plate')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(cropped, cmap='gray')
plt.title('Cropped Plate Area')
plt.axis('off')
plt.show()
if result_text:
    print("✅ Detected Text:")
    for detection in result_text:
        print("-", detection[1])
else:
    print("❌ No text detected.")
