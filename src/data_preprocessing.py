import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images(folder):
    images = []
    labels = []
    for label, subfolder in enumerate(["good_chips", "defective_chips"]):
        path = os.path.join(folder, subfolder)
        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file))
            img = cv2.resize(img, (128, 128))  # Resize for CNN
            images.append(img)
            labels.append(label)  # 0 = Good, 1 = Defective
    return np.array(images), np.array(labels)

# Example usage
if __name__ == "__main__":
    X, y = load_images("../dataset/")
    print(f"Loaded {len(X)} images.")
