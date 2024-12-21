# Car Damage Detection Project

This project is a comprehensive pipeline for detecting car damages using a trained YOLOv11 model. It includes:

1. **A GUI application** for user-friendly detection of car damages on uploaded images.
2. **A Uvicorn server** to serve the model's inference via an API endpoint.
3. **A Jupyter Notebook** for comparing various experiments, visualizing results, and plotting graphs for metrics like precision, recall, mAP\@0.5, and confusion matrices.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [How It Works](#how-it-works)
3. [Setup Instructions](#setup-instructions)
4. [Running the GUI Application](#running-the-gui-application)
5. [Running the Uvicorn Server](#running-the-uvicorn-server)
6. [Using the Compare Notebook](#using-the-compare-notebook)

---

## Project Structure

```
project-directory/
├── gui.py                     # GUI application code
│                  
│   
├── Server/                      # Uvicorn server implementation
│   ├── main.py                  # Main script for the API server
│   ├── requirements.txt         # Python dependencies for the server
├   |─ Models/                      # YOLO models and weights
├── Notebooks/                   # Analysis and comparison notebooks
│   ├── compare.ipynb # Jupyter Notebook for graph generation and comparisons

│                                # Trained YOLOv11 model
├── README.md                    # Documentation (this file)
```

---

## How It Works

### 1. GUI Application

The GUI allows users to upload images and view the detection results directly on the interface. It uses the YOLOv11 model to infer car damages, and the results are displayed with bounding boxes and class labels.

### 2. Uvicorn Server

The Uvicorn server(fastapi) provides an API endpoint to upload images and get JSON responses containing detection results. This is useful for integrating with other systems or for batch processing.

### 3. Jupyter Notebook

The `compare_experiments.ipynb` notebook includes all the metrics and graphs comparing different versions of the YOLOv11 model, such as:

- **Precision, Recall, and mAP\@0.5**
- **Confusion Matrices**
- **Effect of Augmentation (with/without)**

This notebook is designed to help you analyze model performance across experiments.

---

## Setup Instructions

### Prerequisites

- Python 3.8 or above
- pip
- Jupyter Notebook for the analysis notebook


### Install Dependencies

#### GUI Application

```bash
cd app
pip install -r requirements.txt
```

#### Uvicorn Server

```bash
cd Server
pip install -r requirements.txt
```

#### Notebook

Install Jupyter and additional Python libraries:

```bash
pip install jupyter matplotlib pandas numpy
```

---

## Running the GUI Application

1. Navigate to the GUI directory:
   ```bash
   cd app
   ```
2. Run the application:
   ```bash
   python gui.py
   ```

4. Upload an image and press process image.

---

## Running the Uvicorn Server

1. Navigate to the Server directory:
   ```bash
   cd app/api
   ```
2. Run the server:
   ```bash
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## Using the Compare Notebook


1. Review the graphs to analyze the performance of:

   - YOLOv11 Large vs Medium models
   - Models trained with and without data augmentation
   - Effects of augmentation techniques (e.g., flipping, brightness adjustment, rotation)

---

## Notes

- test images are included for testing

## Future Improvements

- Add support for real-time video detection in the GUI.
- Expand API functionality to handle batch image processing.


make 
