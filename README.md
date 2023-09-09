# YOLOv5 Trainer

The YOLOv5 Trainer is a Python program designed to streamline the process of training YOLOv5 models on custom datasets. 
It provides a easy way to model training, testing, and model management.

## Why?
1. I've always found difficulty to follow some training tutorials for yolo. 
2. This is supposed to help the beginners to train faster their models. You just download a dataset (or make your own), then run the trainer.
3. It runs locally. That means you don't need internet or colab to train your models. Just a GPU.
4. Because I'm using it to train some models.

## Features

- **Easy Training**: You can start training right after downloading a dataset in the following format:

    ```
    dataset (root)
    |- test (folder)
    |   |- labels (folder)
    |   |- images (folder)
    |- valid (folder)
    |   |- labels (folder)
    |   |- images (folder)
    |- train (folder)
    |   |- labels (folder)
    |   |- images (folder)
    |- data.yaml
    ```
    - **TIP**: You can download datasets from sites like Roboflow, or prepare your own dataset following the structure mentioned above.


- **Interactive Menu**: After training the model, a user-friendly menu allows you to perform various tasks such as copying the result folder, copying only the model, testing the model, and generating another model.

- **Real-time Training**: The program spawns a separate process to train the YOLOv5 model and displays training progress in real-time.

- **Model Testing**: Test your trained model on custom images and visualize detection results.

## Prerequisites

Before using the YOLOv5 Trainer, ensure you have the following dependencies installed:

- Python (3.6 or later)
- PyTorch (install through the PyTorch website to install CUDA support, if available)
- OpenCv-Python (cv2)
- PyYAML

## Getting Started

1. **Clone this repository:**

   ```bash
   git clone https://github.com/yourusername/yolov5-trainer.git
   cd yolov5-trainer
   ```
2. **Instal requirements:**
   ```bash
   pip install requirements.txt
   ```
3. **Run the trainer:**
   ```bash
   python yolo_trainer.py
   ```
   
4. **Follow the trainer directions**.
5. _**When finished, you can copy the entire folder of the model, or just the model, and test it.**_

## Helping
**_If you liked this repo, you can help me improve it. Just open a PR and I'll be looking into it._**
