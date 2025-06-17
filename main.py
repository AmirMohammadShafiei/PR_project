import cv2
import matplotlib.pyplot as plt
import easyocr
from ultralytics import YOLO
IMAGE_PATH = 'Cars201.png'
MODEL_PATH = 'YOLOv8m_Iran_license_plate_detection.pt'
image = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
model = YOLO(MODEL_PATH)
results = model(image)
plate_found = False
cropped = None
for r in results:
    for box in r.boxes:
        conf = float(box.conf[0])
        if conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = gray[y1:y2, x1:x2]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 3)
            plate_found = True
            break
    if plate_found:
        break
if not plate_found or cropped is None:
    print(" License plate not detected.")
    exit()
reader = easyocr.Reader(['en'])
result_text = reader.readtext(cropped)
plt.figure(figsize=(10,7))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Plate')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(cropped, cmap='gray')
plt.title('Cropped Plate')
plt.axis('off')
plt.show()
if result_text:
    print("✅ Detected Text:")
    for detection in result_text:
        print("-", detection[1])
else:
    print("❌ No text detected.")













