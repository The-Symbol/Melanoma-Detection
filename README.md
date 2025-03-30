# 🩺 Melanoma Detection using CNN

![Melanoma Detection](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Melanoma.jpg/800px-Melanoma.jpg)

## 📌 Overview
This project is a **Convolutional Neural Network (CNN) model** built using **Keras** for detecting melanoma skin cancer from images. The model is trained on a dataset of skin lesion images and classifies them into melanoma and non-melanoma categories.

## 🚀 Features
✅ Uses **CNN with multiple convolutional layers** for feature extraction.
✅ **Batch normalization & dropout layers** to improve generalization.
✅ **Image augmentation** using `ImageDataGenerator`.
✅ **Categorical classification** using softmax activation.
✅ **Trained using categorical cross-entropy loss and Adam optimizer.**

## 📂 Dataset Structure
The dataset consists of images organized into two folders:
```
data/
├── train/
│   ├── melanoma/
│   ├── non-melanoma/
├── test/
│   ├── melanoma/
│   ├── non-melanoma/
```

## 🏗 Model Architecture
- **Conv2D layers** with ReLU activation for feature extraction.
- **MaxPooling layers** to downsample feature maps.
- **Dropout layers** to reduce overfitting.
- **Fully connected (Dense) layers** for classification.
- **Softmax activation** in the output layer for two-class classification.

## 📦 Dependencies
Install required dependencies before running the project:
```bash
pip install numpy pandas opencv-python keras tensorflow
```

## ▶ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/melanoma-detection.git
   cd melanoma-detection
   ```
2. Place the dataset in the `data/train/` and `data/test/` folders.
3. Run the script to train the model:
   ```bash
   python melanoma_detection.py
   ```
4. The model will train for **100 epochs** with training & validation steps.

## ⚙ Model Training Parameters
- **Batch size (Training):** 15
- **Batch size (Testing & Validation):** 2
- **Epochs:** 100
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy

## 📈 Future Improvements
🔹 Use **pretrained models** like VGG16, ResNet for better accuracy.
🔹 Implement **Grad-CAM** to visualize important features in predictions.
🔹 Deploy the model using **Flask or FastAPI**.

## 🤝 Contributing
Contributions are welcome! Feel free to **fork** this repository and submit a **pull request**.

## 📜 License
This project is open-source and available under the MIT License.

---

👨‍💻 **Author:** Akash Krishna

⭐ If you found this project useful, please consider giving it a star! ⭐


