📌 License Plate Detection and Recognition (Iranian & Foreign)
This project detects and recognizes license plates from images using YOLOv8 and OCR libraries. It supports both Persian (Iranian) and English (foreign) plates using EasyOCR and Tesseract respectively.


⚙️ Features


Detect license plates using pretrained YOLOv8 model (YOLOv8m_Iran_license_plate_detection.pt)



OCR for Persian text using EasyOCR



OCR for English text using Tesseract



Image preprocessing and noise removal for improved accuracy



Visualization using matplotlib



Modular code (3 separate scripts for different methods)





🗂️ Project Structure

├── YOLOv8m_Iran_license_plate_detection.pt


│   


├── Cars201.png


│   1.jpg


│   


├── main.py


├── main2.py


├── main3.py


├── README.md


🚀 Installation


Python Version: 3.10 or higher recommended


git clone https://github.com/yourusername/plate-recognition-yolo.git




pip install -r requirements.txt




Install Tesseract:


Windows: [Download](https://github.com/tesseract-ocr/tesseract/wiki)



After installation, add Tesseract path to the script:



pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'




Linux:


sudo apt install tesseract-ocr




🧠 How It Works


There are three different methods used in this project:



YOLOv8 + EasyOCR:



Detects the license plate with YOLO



Crops and feeds it to EasyOCR for Persian recognition



Canny Edge + EasyOCR:



Finds contours with OpenCV (Canny)



Crops the largest rectangular region



OCR with EasyOCR



YOLOv8 + Tesseract:



Plate detection via YOLO



Preprocesses image (resize, threshold, invert)



OCR with Tesseract (English)



🖼️ How to Run

Method 1: YOLO + EasyOCR


main.py


Method 2: Canny + EasyOCR


main2.py


Method 3: YOLO + Tesseract

main3.py




🔤 Language Support


EasyOCR: Persian license plates (fa)



Tesseract: English license plates (eng)



🧪 Sample Output

✅ Detected Text (EasyOCR):
- ۱۲الف۳۴۵

✅ Detected Text (Tesseract):
- XYZ1234


📌 Notes
For best results, use high-resolution images.

Make sure to place the model (YOLOv8m_Iran_license_plate_detection.pt) in the correct directory.

Scripts are modular and can be extended easily.


🙌 Acknowledgments
https://github.com/ultralytics/ultralytics
https://github.com/JaidedAI/EasyOCR
https://github.com/tesseract-ocr/tesseract
https://roboflow.com/
https://huggingface.co/

