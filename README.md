
# Ship Detection using Deep Learning

A comprehensive deep learning project for ship detection in satellite/aerial imagery using Convolutional Neural Networks (CNN) and Transfer Learning techniques.

## 📋 Project Overview

This project implements two different deep learning approaches for ship detection:
1. **Custom CNN from Scratch** - A custom-built convolutional neural network
2. **Transfer Learning with ResNet** - A ResNet-based architecture using transfer learning

The models are trained on the ShipsNet dataset and can detect ships in both individual images and larger scene images using a sliding window approach.

## 🎯 Key Features

- **Binary Classification**: Detects ships vs. non-ships in 80x80 pixel images
- **High Accuracy**: Achieved 96.6% accuracy with custom CNN and 98.1% with ResNet
- **Scene Detection**: Implements sliding window technique for detecting ships in larger satellite images
- **Data Augmentation**: Uses image augmentation to improve model generalization
- **Comprehensive Analysis**: Includes data visualization, confusion matrices, and performance metrics

## 📊 Dataset

- **Total Images**: 4,000
- **Ship Images**: 3,000
- **Non-Ship Images**: 1,000
- **Image Size**: 80x80 pixels
- **Channels**: RGB (3 channels)

## 🏗️ Model Architectures

### 1. Custom CNN (Scratch)
- 4 Convolutional layers with MaxPooling
- Dropout layers for regularization
- Dense layers for classification
- **Test Accuracy**: 96.6%

### 2. ResNet-based (Transfer Learning)
- ResNet blocks with skip connections
- Batch Normalization
- Residual connections for deeper networks
- **Test Accuracy**: 98.1%

## 🛠️ Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Data splitting and metrics
- **PIL (Pillow)**: Image processing
- **imgaug**: Image augmentation

## 📁 Project Structure

```
DMML 2/
├── Scratch CNN.ipynb          # Custom CNN implementation
├── Transfer_Learning.ipynb    # ResNet transfer learning implementation
├── shipsnet/                  # Dataset directory
├── scenes/                    # Scene images for detection
└── README.md                  # Project documentation
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow imgaug
```

### Usage

1. **Load the dataset**:
   ```python
   with open('shipsnet/shipsnet.json') as data_file:
       dataset = json.load(data_file)
   ```

2. **Train the Custom CNN model**:
   - Open `Scratch CNN.ipynb`
   - Run all cells to train and evaluate the model

3. **Train the ResNet model**:
   - Open `Transfer_Learning.ipynb`
   - Run all cells to train and evaluate the model

4. **Detect ships in scene images**:
   - The notebooks include code for sliding window detection
   - Processes larger images by extracting 80x80 patches
   - Visualizes detected ships with bounding boxes

## 📈 Results

### Custom CNN Performance
- **Training Accuracy**: ~94.6%
- **Test Accuracy**: 96.6%
- **Validation Loss**: 0.1028

### ResNet Performance
- **Training Accuracy**: ~97.0%
- **Test Accuracy**: 98.1%
- **Validation Loss**: 0.0419

## 🔍 Key Techniques

1. **Data Preprocessing**:
   - Image normalization (pixel values / 255)
   - Train/validation/test split (60%/20%/20%)
   - One-hot encoding for labels

2. **Data Augmentation**:
   - Rotation (25 degrees)
   - Width/height shifts (10%)
   - Shear transformations (20%)
   - Zoom (20%)
   - Horizontal flipping

3. **Model Training**:
   - Learning rate scheduling
   - Adam optimizer
   - Categorical cross-entropy loss
   - Early stopping considerations

4. **Detection Pipeline**:
   - Sliding window approach for scene images
   - Confidence threshold filtering (0.90)
   - Non-maximum suppression to avoid duplicate detections

## 📊 Visualizations

The project includes:
- Sample images from both classes
- RGB channel analysis
- Pixel intensity histograms
- Training/validation accuracy and loss curves
- Confusion matrices
- Detected ships visualization in scene images

## 🎓 Learning Outcomes

- Implementation of CNN architectures from scratch
- Transfer learning with pre-trained models
- Image classification and object detection
- Data augmentation techniques
- Model evaluation and visualization
- Real-world application to satellite imagery

## 📝 Notes

- The dataset is imbalanced (3:1 ratio), which was handled through data augmentation
- Both models show excellent performance, with ResNet slightly outperforming the custom CNN
- The sliding window approach allows detection in images of any size
- Models can be further improved with more data and hyperparameter tuning

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements.

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

Deep Learning and Machine Learning Project - Ship Detection System

---

**Keywords**: Deep Learning, CNN, ResNet, Transfer Learning, Ship Detection, Computer Vision, Image Classification, Satellite Imagery, TensorFlow, Keras
