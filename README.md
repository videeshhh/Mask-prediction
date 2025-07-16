# Mask Detection using OpenCV & TensorFlow

This project performs real-time face mask detection using a Convolutional Neural Network (CNN) built with TensorFlow and a video feed handled by OpenCV. It can identify whether a person is wearing a mask or not.

## Features

- Real-time mask detection via webcam
- Custom CNN model trained on a labeled face mask dataset
- Uses OpenCV for face detection and live video processing
- Easy to use and extend

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- scikit-learn
- Matplotlib (for plotting training history)

## Project Structure

```
mask-detector/
│
├── dataset/                     # Training images categorized into:
│   ├── with_mask/
│   └── without_mask/
│
├── model/                       # Saved model (e.g., .h5 file)
│
├── train_model.py              # Script to train the CNN model
├── mask_detector.py            # Real-time mask detection using webcam
├── utils.py                    # Utility functions (optional)
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview
```

## Dataset

Use any publicly available face mask dataset or create your own. A common structure:

- `dataset/with_mask/` — Images of people wearing masks
- `dataset/without_mask/` — Images of people without masks

Example: [Kaggle Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mask-detector.git
cd mask-detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install tensorflow opencv-python numpy scikit-learn matplotlib
```

The trained model will be saved in the `model/` directory.

### 3. Run Real-Time Mask Detection

```bash
python nom.ipynb
```

A webcam window will open and display bounding boxes with labels like `Mask` or `No Mask`.

## Model Overview

- Type: Convolutional Neural Network (CNN)
- Layers: Conv2D, MaxPooling2D, Flatten, Dense, Dropout
- Loss Function: Binary Crossentropy
- Metrics: Accuracy
- Accuracy: ~95% on validation set (varies with dataset)

## Notes

- Make sure your webcam is enabled and accessible.
- Detection performance may vary based on lighting and camera quality.
- You can improve accuracy using data augmentation or transfer learning (e.g., MobileNetV2).

## License

This project is licensed under the MIT License.

## Acknowledgments

- Datasets from Kaggle and other open sources
- TensorFlow, OpenCV, and community tutorials
