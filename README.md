# 🔍 Bar Chip Inspection with Machine Learning

This project aims to develop a **machine learning-based visual inspection system** for bar chips before packaging. The system detects defects such as **scratches, misalignment, surface cracks, and contamination** using a **Convolutional Neural Network (CNN)**.

---

## 📁 Project Structure

```
│── dataset/                   # Folder for training images
│   ├── good_chips/            # Good bar chip images
│   ├── defective_chips/       # Defective bar chip images
│── models/                    # Folder to store trained models
│   ├── defect_detector.h5     # Saved TensorFlow model
│── src/                       # Source code
│   ├── data_preprocessing.py  # Preprocess images
│   ├── train_model.py         # Train defect detection model
│   ├── test_model.py          # Test on new images
│   ├── deploy_model.py        # Deploy trained model
│── utils/                     # Helper functions
│   ├── visualization.py       # For plotting images & results
│── notebooks/                 # Jupyter notebooks (optional for testing)
│   ├── data_exploration.ipynb # Notebook for data visualization
│── requirements.txt           # List of dependencies
│── README.md                  # Project documentation

```

---

## 🚀 **Setup Instructions**
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Opengundumstyle/bar_chip_inspection.ai.git
cd bar_chip_inspection
```
### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Prepare the Dataset
Place images in dataset/good_chips/ and dataset/defective_chips/.
Use LabelImg for manual defect annotation (optional).

### 4️⃣ Train the Model
python src/train_model.py

### 5️⃣ Test on New Images
python src/test_model.py --image_path "path/to/test_image.jpg"

🤖 Technology Stack
Python 3.9+
TensorFlow/Keras (for deep learning)
OpenCV (for image processing)
Scikit-learn (for classical ML)
Matplotlib (for visualization)
Jupyter Notebooks (for testing & data analysis)

🎯 Goals
✔ Develop a customized magnified workstation or integrate with IOA (Integrated Optical Assembly)
✔ Train a CNN model for defect detection
✔ Optimize for real-time processing on production lines
✔ Deploy on Edge AI hardware (Jetson/Raspberry Pi)

🛠 Next Steps
✅ Collect & preprocess bar chip images
✅ Train an initial defect detection model
🔲 Optimize accuracy with data augmentation
🔲 Deploy for real-world testing

✨ Contributing
Fork this repository.
Create a feature branch (git checkout -b feature-branch).
Commit your changes (git commit -m "Added new feature").
Push to GitHub (git push origin feature-branch).
Open a Pull Request!

📜 License
MIT License © 2025 Opengundumstyle