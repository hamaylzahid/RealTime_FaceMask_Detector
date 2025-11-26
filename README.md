<h1 align="center">Real-Time Face Mask Detector</h1><br>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11-blue" alt="Python Version"></a>
  <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/tensorflow-2.15.0-orange" alt="TensorFlow Version"></a>
  <a href="https://opencv.org/"><img src="https://img.shields.io/badge/opencv-4.12.0-brightgreen" alt="OpenCV Version"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.7.0%2Bcpu-red" alt="PyTorch Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

<p align="center">
  <strong>Detects face masks in real-time using computer vision and deep learning.</strong><br>
  Built for efficient and accurate mask detection in video streams, this system leverages <strong>TensorFlow, OpenCV, and PyTorch</strong> for preprocessing, face detection, and prediction.
</p>

<p align="center">
  Key Features:
  <ul style="list-style-type:none; text-align:center; padding-left:0;">
    <li>Real-time detection using webcam input</li>
    <li>Pretrained deep learning model for mask/no-mask classification</li>
    <li>Accurate face detection with Haar Cascades</li>
    <li>FPS counter for performance monitoring</li>
    <li>Easy-to-use Python implementation</li>
  </ul>
</p>


<hr>

<br><h2  align="center">Table of Contents</h2><br>
<ul>
  <li><a href="#project-overview">Project Overview</a></li>
  <li><a href="#features">Features</a></li>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#folder-structure">Folder Structure</a></li>
  <li><a href="#dependencies">Dependencies</a></li>
  <li><a href="#license">License</a></li>
  <li><a href="#notes">Notes</a></li>
</ul>

<hr>

<br><h2  align="center" id="project-overview">Project Overview</h2><br>
<p>This project implements a <strong>real-time face mask detection system</strong> using deep learning. 
It uses a <strong>pretrained Keras/TensorFlow model</strong> to classify faces as <em>Mask</em> or <em>No Mask</em> 
and highlights detected faces in the video feed with colored bounding boxes.</p>

<ul>
  <li>Detects multiple faces in real-time using OpenCV.</li>
  <li>Displays detection labels and colored bounding boxes for Mask/No Mask.</li>
  <li>Can be extended for CCTV monitoring or automated entry systems.</li>
</ul>

<br><h2  align="center" id="features">Features</h2><br>
<ul>
  <li>Real-time face detection using Haar Cascades.</li>
  <li>Preprocessing pipeline for face ROI.</li>
  <li>Mask detection with a trained CNN model.</li>
  <li>Optional FPS counter for performance monitoring.</li>
  <li>Easy-to-use Python script or Jupyter Notebook.</li>
</ul>

<br><h2  align="center" id="installation">Installation</h2><br>
<pre><code># Clone repository
git clone https://github.com/hamaylzahid/RealTime_FaceMask_Detector.git
cd RealTime_FaceMask_Detector

### Create virtual environment (recommended)
python -m venv mask_env
mask_env\Scripts\activate   # Windows
source mask_env/bin/activate # macOS/Linux

### Install dependencies
pip install -r requirements.txt

### Run Jupyter Notebook
jupyter notebook RealTime_FaceMask_Detector.ipynb
</code></pre>

<br><h2  align="center" id="usage">Usage (Python Script Example)</h2><br>
<pre><code>import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

mask_model = load_model("mask_detector.model")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_face(face_img):
    face = cv2.resize(face_img, (224, 224))
    face = img_to_array(face)
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
    return face

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_input = preprocess_face(face_img)
        pred = mask_model.predict(face_input)[0]
        mask_prob, no_mask_prob = pred
        label = "Mask" if mask_prob > no_mask_prob else "No Mask"
        color = (0, 255, 0) if mask_prob > no_mask_prob else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
</code></pre>

<br><h2  align="center" id="folder-structure">Folder Structure</h2><br>
<pre><code>
RealTime_FaceMask_Detector/
│
├── RealTime_FaceMask_Detector.ipynb
├── mask_detector.model
├── requirements.txt
|── README.md

</code></pre>

<br><h2 align="center" id="dependencies">Dependencies</h2><br>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python 3.11">
  </a>
  <a href="https://www.tensorflow.org/">
    <img src="https://img.shields.io/badge/TensorFlow-2.15.0-orange" alt="TensorFlow 2.15.0">
  </a>
  <a href="https://keras.io/">
    <img src="https://img.shields.io/badge/Keras-Latest-red" alt="Keras">
  </a>
  <a href="https://opencv.org/">
    <img src="https://img.shields.io/badge/OpenCV-4.12.0-brightgreen" alt="OpenCV 4.12.0">
  </a>
  <a href="https://numpy.org/">
    <img src="https://img.shields.io/badge/NumPy-1.25.x-blueviolet" alt="NumPy 1.25.x">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.7.0%2Bcpu-red" alt="PyTorch 2.7.0+cpu">
  </a>
</p>



<br><h2  align="center" id="notes">Notes</h2>
<ul><br>
  <li>Ensure your virtual environment uses Python 3.11 for full compatibility.</li>
  <li>This system can be extended for SSD or DNN-based face detectors for higher accuracy.</li>
  <li>FPS counter can be added to monitor real-time performance.</li>
</ul>
<hr>

<br><h2 align="center">Contact & Contribution</h2><br>

<p align="center">
  <em>Have feedback, want to collaborate, or just say hello?</em><br>
  <strong>Let’s connect and build something useful together.</strong>
</p>

<p align="center">
  <a href="mailto:maylzahid588@gmail.com">maylzahid588@gmail.com</a> &nbsp; | &nbsp;
  <a href="https://www.linkedin.com/in/hamaylzahid/">LinkedIn Profile</a> &nbsp; | &nbsp;
  <a href="https://github.com/hamaylzahid/RealTime_FaceMask_Detector">GitHub Repo</a>
</p>

<p align="center">
  <a href="https://github.com/hamaylzahid/RealTime_FaceMask_Detector/stargazers">
    <img src="https://img.shields.io/github/stars/hamaylzahid/RealTime_FaceMask_Detector?style=for-the-badge&logo=github&color=yellow" alt="Star Badge" />
  </a>
  <a href="https://github.com/hamaylzahid/RealTime_FaceMask_Detector/pulls">
    <img src="https://img.shields.io/badge/Contribute-Pull%20Requests%20Welcome-2ea44f?style=for-the-badge&logo=github" alt="PRs Welcome Badge" />
  </a>
</p>

<p align="center">
  Found this project useful? Give it a star on GitHub.<br>
  Want to improve it? Submit a Pull Request and contribute to enhancing this system.
</p>

<p align="center">
  <sub><i>Your ideas and contributions help improve this real-time AI system.</i></sub>
</p>

<hr>

<br><h2 align="center">License</h2><br>

<p align="center">
  <a href="https://github.com/hamaylzahid/RealTime_FaceMask_Detector/commits/main">
    <img src="https://img.shields.io/github/last-commit/hamaylzahid/RealTime_FaceMask_Detector?color=blue" alt="Last Commit">
  </a>
  <a href="https://github.com/hamaylzahid/RealTime_FaceMask_Detector">
    <img src="https://img.shields.io/github/repo-size/hamaylzahid/RealTime_FaceMask_Detector?color=lightgrey" alt="Repo Size">
  </a>
</p>

<p align="center">
  This project is licensed under the <strong>MIT License</strong> — open to use, customize, and evolve.
</p>

<p align="center">
  <a href="https://img.shields.io/badge/Project-Status-Complete-brightgreen" alt="Project Status"></a>
  <a href="https://img.shields.io/badge/License-MIT-blue" alt="License"></a>
</p>

<p align="center">
  Use this project to showcase your skills in real-time computer vision and AI.<br>
  Clone, modify, and improve it for practical applications.
</p>

<p align="center">
  <a href="https://github.com/hamaylzahid">
    <img src="https://img.shields.io/badge/GitHub-%40hamaylzahid-181717?style=flat-square&logo=github" alt="GitHub" />
  </a>
  •
  <a href="mailto:maylzahid588@gmail.com">
    <img src="https://img.shields.io/badge/Email-Contact%20Me-red?style=flat-square&logo=gmail&logoColor=white" alt="Email" />
  </a>
  •
  <a href="https://github.com/hamaylzahid/RealTime_FaceMask_Detector">
    <img src="https://img.shields.io/badge/Repo-Link-blueviolet?style=flat-square&logo=github" alt="Repo" />
  </a>
  <br>
  <a href="https://github.com/hamaylzahid/RealTime_FaceMask_Detector/fork">
    <img src="https://img.shields.io/badge/Fork%20This%20Project-Contribute-2ea44f?style=flat-square&logo=github" alt="Fork Badge" />
  </a>
</p>

<p align="center">
  <sub><i>Inspired by real-time AI development. Built with code and care.</i></sub>
</p>

<p align="center">
  Use this project to showcase your AI and computer vision skills.<br>
  Clone it, modify it, and expand it to improve real-time face mask detection systems.
</p>

