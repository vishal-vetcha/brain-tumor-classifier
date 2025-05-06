# Brain Tumor Detection and Classification from MRI Scans

## Overview

This project implements a deep learning model based on Convolutional Neural Networks (CNNs) to classify brain MRI images into four categories: glioma, meningioma, no tumor, and pituitary tumor. It leverages the PyTorch deep learning framework for model building and training, and OpenCV for essential image preprocessing steps. The goal is to create an accurate and user-friendly tool for assisting in the diagnosis of brain tumors from MRI scans.

## Dataset

This project utilizes a combined dataset of 7023 brain MRI images, sourced from figshare, the SARTAJ dataset, and the Br35H dataset, as described in the original Kaggle dataset: [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). The dataset is organized into four classes:

* **Glioma:** Tumors originating in the glial cells of the brain.
* **Meningioma:** Tumors arising from the meninges surrounding the brain and spinal cord.
* **No Tumor:** MRI scans showing no presence of a brain tumor.
* **Pituitary Tumor:** Tumors developing in the pituitary gland.

**Note:** The dataset contains images of varying sizes. Preprocessing steps, including resizing and margin removal, are crucial for optimal model performance. The project addresses a known issue with the glioma class in the original SARTAJ dataset by utilizing corrected images from the figshare source.

## Project Structure
brain_tumor_classifier/
├── data/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── no_tumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── no_tumor/
│       └── pituitary/
├── src/
│   ├── preprocess.py         # Image preprocessing using OpenCV
│   ├── model.py              # PyTorch CNN model definition
│   ├── train.py              # Script for training the CNN model
│   ├── predict.py            # Script for making predictions on single images
│   └── utils.py              # Helper functions
├── user_interface.py         # User interface for uploading and predicting images
├── requirements.txt          # List of Python dependencies
└── README.md

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd brain_tumor_classifier
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    pip install -r requirements.txt
    ```
    *(You will need to generate the `requirements.txt` file using `pip freeze > requirements.txt` after installing the necessary libraries.)*

## Usage

1.  **Ensure the dataset is organized** in the `data/Training` and `data/Testing` directories with subfolders for each class (glioma, meningioma, no_tumor, pituitary).

2.  **Run the training script:**
    ```bash
    python src/train.py --data_dir path/to/your/data --epochs [number_of_epochs] --batch_size [batch_size]
    ```
    *(Adjust the `data_dir`, `epochs`, and `batch_size` arguments as needed.)*

3.  **Run the prediction script for a single image:**
    ```bash
    python src/predict.py --model_path path/to/your/trained_model.pth --image_path path/to/your/test_image.jpg
    ```
    *(Replace `path/to/your/trained_model.pth` with the path to your trained model weights and `path/to/your/test_image.jpg` with the path to the image you want to predict.)*

4.  **Run the user interface:**
    ```bash
    python user_interface.py
    ```
    *(Follow the instructions provided by the user interface (likely a local web address) to upload an MRI image and get the prediction.)*

## Image Preprocessing

The `src/preprocess.py` script implements the following image preprocessing techniques using OpenCV:

* **Resizing:** Images are resized to a consistent dimension to be compatible with the CNN input layer.
* **Noise Reduction:** Techniques like Gaussian blurring may be applied to reduce noise.
* **Normalization:** Pixel values are normalized to a specific range (e.g., 0 to 1) to improve training stability.
* **Contrast Enhancement:** Methods like histogram equalization or adaptive histogram equalization might be used to enhance image contrast.
* **Margin Removal:** Extra black margins around the brain scans are removed to focus on the region of interest.

## Model Architecture

The brain tumor classification model is a Convolutional Neural Network (CNN) built using the PyTorch library. The architecture consists of convolutional layers, pooling layers, activation functions (ReLU), batch normalization, dropout, and fully connected layers leading to a final output layer with four neurons (corresponding to the four classes). The specific architecture details are defined in `src/model.py`.

## Evaluation

The model's performance is evaluated on the testing dataset using metrics such as:

* **Accuracy:** The overall percentage of correctly classified images.
* **Precision:** The ability of the model to correctly identify positive instances for each class.
* **Recall:** The ability of the model to find all the positive instances for each class.
* **F1-Score:** The harmonic mean of precision and recall.
* **Confusion Matrix:** A table showing the distribution of predicted and actual classes.

## Contributing

Contributions to this project are welcome. Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-contribution`).
3.  Make your changes.
4.  Commit your changes (`git commit -m "Add your contribution"`).
5.  Push to the branch (`git push origin feature/your-contribution`).
6.  Create a new Pull Request.

## License

[Specify your license here, e.g., MIT License]

---

This detailed plan and README should provide a solid foundation for your brain tumor detection project. Let me know if you have any specific questions or would like to delve deeper into any of these steps!