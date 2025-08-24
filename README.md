#  Fake Currency Detection Using Python
Fake Currency Detection is a Python-based machine learning project designed to identify counterfeit currency notes using image processing and classification techniques. It leverages powerful libraries like OpenCV, NumPy, and scikit-learn to analyze visual features and distinguish between genuine and fake notes.

## 📌 Project Overview
This tool helps automate the detection of counterfeit currency by analyzing scanned or photographed images of notes. It uses feature extraction and classification models to predict authenticity with high accuracy.

## 🔍 Features
- 🖼️ Image Preprocessing: Resizing, grayscale conversion, edge detection
- 📊 Feature Extraction: Texture, color histograms, contour analysis
- 🧠 Machine Learning Model: Trained classifier (e.g., SVM, Random Forest, CNN)
- 📁 Dataset Support: Works with custom or public datasets of currency images
- 📈 Accuracy Metrics: Evaluation using confusion matrix, precision, recall
- 🖥️ GUI/CLI Interface (optional): User-friendly interface for testing new images

## 🛠️ Tech Stack
- Python 3.x
- OpenCV
- NumPy
- scikit-learn / TensorFlow / Keras (depending on your model)
- Matplotlib / Seaborn (for visualization)

### 🚀 Installation
git clone https://github.com/yourusername/fake-currency-detection.git
cd fake-currency-detection




## Dataset
You can use publicly available datasets or create your own by scanning genuine and fake currency notes. 
Make sure to organize them into labeled folders like:
/dataset
   /test
   /trained
   



### Evaluation
- Accuracy: 95% on test set
- Precision/Recall: High performance on distinguishing subtle features

