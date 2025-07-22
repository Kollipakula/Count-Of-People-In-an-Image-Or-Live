
Dual-Approach Crowd Counting System

This project presents a novel dual-method deep learning technique for crowd counting, capable of analyzing both real-time video streams and static images. The system leverages MTCNN for efficient real-time face detection and a VGG16-based Convolutional Neural Network (CNN) for accurate density map estimation in highly congested scenes. This hybrid approach ensures comprehensive and adaptable crowd monitoring for applications such as public security, event management, and smart city infrastructure.

---

Key Features

* Real-Time Crowd Detection:** Uses Multi-task Cascaded Convolutional Networks (MTCNN) to detect and count faces from live video feeds like webcams and CCTV.
* Static Image Crowd Estimation:** Employs a VGG16-based CNN to generate density heatmaps, accurately estimating crowds in highly congested scenes.
* Dual-Approach Flexibility:** Integrates both real-time and static image-based methods to handle a wide range of crowd analysis scenarios.
* Robust Performance:** A scalable and AI-driven solution overcoming limitations of single-method crowd counting systems.
* Visualized Output:** Provides bounding box overlays with live counts on video and heatmap visualizations with estimated people counts for static images.

---

Proposed Methodology

1. Real-Time Crowd Detection with MTCNN

* Input Capture:** Processes live video feeds.
* Frame Processing:** Converts frames to BGR color space.
* Face Detection:** MTCNN detects faces and returns bounding boxes with confidence scores.
* Filtering and Annotation:** Filters detections based on a confidence threshold (e.g., 90%) and annotates bounding boxes.
* Live Count Display:** Displays the real-time face count on the video feed.

2. Static Image-Based Crowd Estimation using Density Maps

* Dataset Preparation:** Uses the ShanghaiTech Part A dataset with head annotations.
* Preprocessing:** Resizes images (e.g., 512x512) and normalizes density maps.
* Model Input:** Custom CNN with VGG16 backbone generates density maps.
* People Count Calculation:** Sum of pixel values in the predicted density map
  $\text{Count} = \sum_{i=1}^H \sum_{j=1}^W D(i,j)$
* **Visualization:** Applies colormaps like Jet for visual representation.

---

 Model Architecture

### 1. Real-Time Detection Module (MTCNN)

MTCNN operates in three cascaded stages:

* P-Net (Proposal Network):** Proposes candidate face regions.
* R-Net (Refine Network):** Refines candidates and removes false positives.
* O-Net (Output Network):** Performs accurate face localization and landmark detection.


---

## 📊 Experimental Results

### Static Image Evaluation (ShanghaiTech Part A)

* Sample actual count: 431
* Predicted count: **437.40**

### Real-Time Detection Performance

* **Confidence Threshold:** 0.90
* **Frame Size:** 640x480
* **Avg Faces Detected/Frame:** 4–15
* **Latency:** \~0.2 sec/frame
* **Accuracy:** 92.4%
* **False Positive Rate:** 3.1%
* **False Negative Rate:** 4.5%
* **Average Precision (AP):** 91.7%

---

## 🛠 Technologies Used

* Python
* PyTorch
* OpenCV
* MTCNN
* NumPy
* Matplotlib




## 🔭 Future Work

* Integrate attention-based models like CSRNet/MCNN for better density accuracy.
* Optimize for real-time performance on edge devices.
* Develop web/mobile UI for real-world deployment.
* Add features like real-time age/gender estimation.
* Train on datasets like UCF-QNRF or WorldExpo’10 for better generalization.
* Combine people counting with anomaly detection systems.

---

## 🤝 Contributing

We welcome contributions!
Feel free to fork the repo, open issues, or submit pull requests.

---
![Real-Time Detection](images/real_time.png)

---

Let me know if you'd like me to create a `requirements.txt` file or split this into individual markdown sections for your GitHub wiki/docs!
