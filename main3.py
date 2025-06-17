import cv2
import pytesseract
from ultralytics import YOLO
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
IMAGE_PATH = 'Cars276.png'
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
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            plate_found = True
            break
    if plate_found:
        break
if not plate_found or cropped is None:
    print("❌ License plate not detected.")
    exit()
plate_resized = cv2.resize(cropped, (300, 100))
_, plate_thresh = cv2.threshold(plate_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plate_inverted = cv2.bitwise_not(plate_thresh)
custom_config = r'--oem 3 --psm 6 -l eng'
text = pytesseract.image_to_string(plate_inverted, config=custom_config)
def clean_text(t):
    return ''.join([c for c in t if c.isalnum() or c.isspace()]).strip()
cleaned_text = clean_text(text)
plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected License Plate')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(plate_inverted, cmap='gray')
plt.title('Cropped Plate for OCR')
plt.axis('off')
plt.show()
print("\n✅ License Plate Text (Tesseract OCR):")
print(cleaned_text if cleaned_text else "❌ No readable text detected.")
