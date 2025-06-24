# Car Detection and Retrieval Engine

A two-stage deep learning system built with PyTorch to detect vehicles and perform fine-grained classification on common Indonesian car models from video footage.

---

### Table of Contents

- [Project Description](#project-description)
- [Project Architecture](#project-architecture)
- [Installation Guide](#installation-guide)
- [Dataset Information](#dataset-information)
- [Conclusions](#conclusions)
- [Future Works](#future-works)

---

### Project Description

This project was created to address the need for a more intelligent vehicle analysis system tailored for the diverse Indonesian automotive market. Standard object detectors can identify a "car," but they lack the granularity to distinguish between specific models like a 'Toyota Avanza' or a 'Honda Brio'. This system bridges that gap by implementing a pipeline that not only detects vehicles in traffic but also identifies their specific model type, enabling more advanced applications in traffic analysis, vehicle retrieval, and market research.

The system is built on a two-stage deep learning pipeline. The first stage uses a **Single Shot MultiBox Detector (SSD300)** with a VGG16 backbone to perform robust, general vehicle detection. Once a car is located, the cropped image is passed to the second stage: a **VGG16 classification model** trained from scratch on a dataset of 13 common Indonesian car models. This modular design allows each model to be optimized for its specific task, creating an end-to-end system for fine-grained car retrieval.

### Project Architecture

This repository follows a structured layout to separate data, notebooks, source code, and final outputs.

```
.
├── data
│   ├── object_classification
│   └── object_detection
├── models
│   └── logs
├── notebooks
│   ├── 1_data_preparation
│   ├── 2_object_detection_model_training_and_evaluation
│   └── 3_object_classification_model_training_and_classification
├── references
├── reports
│   └── figures
└── src
```

### Installation Guide

This project was developed using **Python 3.11.13**.

1.  **Clone the Repository**

    ```bash
    git clone [https://github.com/hardefarogonondo/car-detection-and-retrieval-engine.git](https://github.com/hardefarogonondo/car-detection-and-retrieval-engine.git)
    cd car-detection-and-retrieval-engine
    ```

2.  **Create and Activate a Virtual Environment** (Recommended)

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install PyTorch**
    This is a critical step. Install PyTorch according to your system's hardware (CPU or specific CUDA version). For other configurations, please visit the [official PyTorch website](https://pytorch.org/get-started/locally/). The command below is an example for a specific CUDA-enabled setup:

    ```bash
    pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
    ```

4.  **Install Project Dependencies**
    Install the core dependencies from the `requirements.txt` file and other necessary libraries.

    ```bash
    pip install -r requirements.txt
    pip install torchmetrics[detection] roboflow
    ```

5.  **Download Model Weights**
    Due to file size limits, some model weights are hosted on Google Drive. Please download them from the link below and place them in the appropriate directories.

    - **Google Drive Link:** [Model Weights & Demo Video](https://drive.google.com/drive/folders/10zfLHBUop1zMpNTSUeojdjQT3WugN0wF?usp=sharing)

6.  **Run the Demo**
    Navigate to the `src` directory and run the main script.
    ```bash
    cd src
    python main.py
    ```
    The script will process the test video and save the annotated output in the `/reports` folder.

### Dataset Information

This project utilizes two distinct datasets sourced from [Roboflow](https://roboflow.com/):

1.  **Object Detection Dataset:** A general vehicle detection dataset provided by user _lynkeus03_. It contains annotations for 7 classes (car, bus, truck, etc.) and was used to train the SSD300 model.
2.  **Car Classification Dataset:** A fine-grained dataset of Indonesian car models provided by user _smarnozzle_. It contains images for 13 different car models and was used to train the VGG16 classification model from scratch.

### Conclusions

- A functional proof-of-concept for a two-stage car retrieval system was successfully developed.
- The SSD300 detection model achieved a test **mAP of 0.1073**.
- The VGG16 classification model achieved a final **test accuracy of 56.06%**.
- The primary performance bottleneck for both models was identified as **overfitting**, where the models learned the training data well but failed to generalize to new data.

### Future Works

- **Combat Overfitting:** Implement transfer learning for the VGG16 classifier and apply aggressive data augmentation and regularization techniques (weight decay, dropout) to both models.
- **Expand Datasets:** Increase the number of images per car model and diversify the datasets with varied lighting, angles, and occlusion conditions.
- **Optimize for Real-Time Performance:** Explore lighter model backbones (e.g., MobileNet) and model optimization techniques like quantization to improve inference speed.
