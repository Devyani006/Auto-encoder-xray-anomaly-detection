# 🩺 X-Ray Anomaly Detection using Deep Learning (Autoencoder + ViT)

<p align="center">
  <b>🚀 GPU-Accelerated Medical Imaging | Unsupervised Learning | Real-World Application</b>
</p>


## 📌 Overview

This project implements a **deep learning-based anomaly detection system** for X-ray images using an **Autoencoder architecture**, enhanced with modern techniques such as **Vision Transformers (ViT)**.

The model learns patterns from *normal medical images* and detects abnormalities using **reconstruction error**, making it highly useful in scenarios where labeled anomaly data is scarce.

---

## 🎯 Key Highlights

* ⚡ **End-to-End ML Pipeline**: Data preprocessing → Model training → Evaluation
* 🧠 **Unsupervised Learning**: No labeled anomalies required
* 🚀 **GPU Acceleration**: Leveraged CUDA for faster training (RTX 3050)
* 🔍 **High Sensitivity**: Detects subtle anomalies via reconstruction loss
* 🧩 **Scalable Design**: Easily extendable to other medical datasets
* 📦 **Production-Aware Setup**: Clean repo structure, external model hosting

---

## 🏗️ Project Architecture

```id="lg4y4z"
Input X-ray Image
        │
        ▼
   Autoencoder Model
        │
        ▼
Reconstructed Image
        │
        ▼
Reconstruction Error
        │
        ▼
 Anomaly Detection
```

---

## ⚙️ Tech Stack

* **Languages**: Python
* **Frameworks**: PyTorch / TensorFlow
* **Libraries**: NumPy, OpenCV, Matplotlib
* **Hardware**: NVIDIA GPU (CUDA-enabled)
* **Tools**: Git, VS Code

---

## 📂 Project Structure

```id="x6axg2"
xray-anomaly-detection/
│── data/                # Dataset (excluded via .gitignore)
│── models/              # Trained models (external storage)
│── src/                 # Core source code
│── notebooks/           # Experiments & analysis
│── requirements.txt     # Dependencies
│── README.md            # Documentation
```

---

## 📦 Model Files (External Hosting)

Due to GitHub file size limits, trained models are hosted on **Kaggle**.

👉 **Download here**:
https://www.kaggle.com/datasets/devyanidhokrat/auto-encoder

📌 After downloading, place inside:

```id="f7c3az"
models/
```

---

## ▶️ How to Run

### 1️⃣ Clone Repository

```id="wd3hgl"
git clone https://github.com/Devyani006/Auto-encoder-xray-anomaly-detection/
cd Auto-encoder-xray-anomaly-detection
```

### 2️⃣ Setup Environment

```id="zcg9mr"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Run Project

```id="3nnn2v"
python main.py
```


## 🧠 Methodology

1. Train Autoencoder on **normal X-ray images**
2. Model learns compressed representation
3. Reconstructs input image
4. Compute **Reconstruction Loss**
5. High loss ⇒ **Anomaly detected**


## 📊 Results & Impact

* ✅ Efficient anomaly detection without labeled abnormal data
* ⚡ Significant training speed-up using GPU acceleration
* 📉 Reduced dependency on manual medical annotations
* 🧪 Suitable for research, prototyping, and healthcare AI systems



## 🔮 Future Scope

* 🧠 Integrate Vision Transformers for better feature extraction
* 🌐 Deploy as a web-based diagnostic tool
* 📊 Improve anomaly scoring with advanced metrics
* 📡 Real-time inference system



## 👩‍💻 Author

**Devyani Dhokrat**

* Backend Developer | AI Enthusiast | Data Science Learner


## 📬 Let’s Connect

If you found this project interesting or want to collaborate, feel free to connect!


## ⭐ Show Your Support

If you like this project, consider giving it a ⭐ on GitHub!
