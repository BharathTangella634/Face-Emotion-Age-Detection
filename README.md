# 🎭 Real-Time Face Emotion & Age Detection

A real-time Face Emotion Recognition and Age Estimation system built using PyTorch and OpenCV.

This project combines:

- 7-Class Emotion Recognition (FER+)
- Age Estimation (UTKFace Dataset)
- Live Camera Support (USB & IP Camera)
- GPU Acceleration (if available)

------------------------------------------------------------

🚀 FEATURES

✔ Emotion Prediction  
✔ Age Estimation  
✔ Bounding Box Detection  
✔ Live Webcam Support  
✔ IP Camera Streaming Support  

------------------------------------------------------------

🧠 TECH STACK

- Python 3.8+
- PyTorch
- OpenCV
- torchvision
- Haar Cascade
- ResNet (Age Model)

------------------------------------------------------------

📁 PROJECT STRUCTURE

Face-Emotion-Age-Detection/
│
├── Facial-Emotion-Recognition-PyTorch-ONNX/
├── Facial_Age_estimation_PyTorch/
├── FER_live_cam.py
├── FER_image.py
├── requirements.txt
└── README.md

------------------------------------------------------------

⚠ MODELS & DATASETS NOT INCLUDED

Due to GitHub file size limits, trained models, CSV files, and datasets are not included in this repository.

Download required files from:

👉 GOOGLE DRIVE LINK:
YOUR_GOOGLE_DRIVE_LINK

------------------------------------------------------------

📦 AFTER DOWNLOADING

Extract project_assets.zip

You will see:

project_assets/
│
├── checkpoints/
├── datasets/
└── csv_files/

------------------------------------------------------------

📂 PLACE FILES IN CORRECT LOCATIONS

1️⃣ AGE DATASET

Move:

datasets/utkcropped/

To:

Facial_Age_estimation_PyTorch/

Final structure:

Facial_Age_estimation_PyTorch/utkcropped/

------------------------------------------------------------

2️⃣ AGE MODEL

Move:

checkpoints/age_model.pt

To:

Facial_Age_estimation_PyTorch/checkpoints/

------------------------------------------------------------

3️⃣ EMOTION MODELS

Move:

checkpoints/best_model.pt  
checkpoints/FER_trained_model.pt  

To:

Facial-Emotion-Recognition-PyTorch-ONNX/PyTorch/

------------------------------------------------------------

4️⃣ CSV FILES

Move:

csv_files/train_ferplus.csv  
csv_files/total_ferplus.csv  
csv_files/train.csv  
csv_files/test.csv  

Back to their respective original directories inside:

Facial-Emotion-Recognition-PyTorch-ONNX/

------------------------------------------------------------

🔧 INSTALLATION

Step 1: Create virtual environment (recommended)

python -m venv venv
source venv/bin/activate

Step 2: Install dependencies

pip install -r requirements.txt

------------------------------------------------------------

▶ RUN ON IMAGE

python FER_image.py --path path_to_image.jpg

------------------------------------------------------------

▶ RUN LIVE CAMERA (USB)

python FER_live_cam.py

------------------------------------------------------------

▶ RUN LIVE CAMERA (IP CAMERA)

Edit inside FER_live_cam.py:

cap = cv2.VideoCapture("http://YOUR_LAPTOP_IP:5000/video")

------------------------------------------------------------

⌨ CONTROLS

Press:

q

To exit camera window.

------------------------------------------------------------

📌 NOTES

- GPU will be used automatically if available.
- Models trained on FER+ and UTKFace datasets.
- Haar Cascade is used for face detection.
- Make sure required files are placed correctly before running.

------------------------------------------------------------

📜 LICENSE

This project is for educational purposes.
