# IMDB Movie Review Sentiment Analysis: Classical ML vs. Deep Learning

This project performs sentiment analysis on the IMDB movie review dataset. It implements and compares the performance of several classical machine learning models (Logistic Regression, Linear SVM, Multinomial Naive Bayes) against a Deep Learning model (CNN + BiLSTM) for binary sentiment classification (positive/negative).

## Table of Contents
1.  [Overview](#overview)
2.  [Dataset](#dataset)
3.  [Models Implemented](#models-implemented)
4.  [Workflow](#workflow)
5.  [Technologies Used](#technologies-used)
6.  [Setup and Installation](#setup-and-installation)
7.  [How to Run](#how-to-run)
8.  [Notebook Structure](#notebook-structure)
9.  [Example Results](#example-results)
10. [License](#license)

## Overview

The primary goal of this project is to build and evaluate different models for classifying movie reviews as either positive or negative. The notebook provides a framework to:
*   Load and preprocess text data.
*   Implement feature engineering techniques suitable for both classical ML and Deep Learning.
*   Train and evaluate various models.
*   Visualize model performance and training history.

## Dataset

The project utilizes the **IMDB Movie Review dataset**, sourced from `tensorflow_datasets`.
*   **Description:** A dataset of 50,000 movie reviews for binary sentiment classification.
*   **Split:** 25,000 reviews for training and 25,000 for testing. Each set contains an equal number of positive and negative reviews.
*   **Source:** `tfds.load('imdb_reviews', split=['train', 'test'], as_supervised=True, with_info=True)`

## Models Implemented

The notebook allows for the selection and training of the following models:

1.  **Classical Machine Learning Models (Scikit-learn):**
    *   Logistic Regression
    *   Linear Support Vector Machine (Linear SVM)
    *   Multinomial Naive Bayes
    *   *Feature Engineering: TF-IDF Vectorization*
    *   *Hyperparameter Tuning: GridSearchCV*

2.  **Deep Learning Model (TensorFlow/Keras):**
    *   Convolutional Neural Network (CNN) + Bidirectional LSTM (BiLSTM)
        *   Embedding Layer
        *   1D Convolutional Layers
        *   MaxPooling Layers
        *   Bidirectional LSTM Layer
        *   Dense Layers with Dropout
        *   Sigmoid Output Layer
    *   *Feature Engineering: Tokenization and Padding*
    *   *Training: EarlyStopping callback*

## Workflow

The notebook follows these general steps:

1.  **Setup & Imports:** Import necessary libraries and configure environment settings (e.g., GPU usage, random seeds).
2.  **Data Loading:** Load the IMDB dataset using TensorFlow Datasets.
3.  **Initial Data Exploration:** Inspect the raw data and its structure.
4.  **Text Preprocessing:**
    *   Define a function to clean text (remove punctuation, convert to lowercase).
    *   Apply preprocessing to training and test sets.
    *   NLTK stopwords are downloaded and can be used for removal (though the notebook notes it isn't always best).
5.  **Data Splitting:** Split the training data into training and validation sets.
6.  **Feature Engineering (Conditional):**
    *   **For Classical ML:** TF-IDF vectorization is applied.
    *   **For Deep Learning:** Keras `Tokenizer` is used for text-to-sequence conversion, followed by padding.
7.  **Model Selection & Definition:**
    *   A `MODEL_CHOICE` variable allows users to select which model family (classical or DL) and specific model to run.
    *   Hyperparameter grids are defined for classical models.
    *   The Keras Sequential API is used to define the CNN+BiLSTM architecture.
8.  **Model Training (Conditional):**
    *   **Classical ML:** `GridSearchCV` is used to find the best hyperparameters and train the model.
    *   **Deep Learning:** The Keras model is compiled and trained using `.fit()`, with an `EarlyStopping` callback to prevent overfitting.
9.  **Evaluation:**
    *   The trained model is evaluated on the validation set and then on the unseen test set.
    *   Metrics: Accuracy, Precision, Recall, F1-Score.
    *   Confusion Matrices are generated and plotted.
10. **Visualization:**
    *   Training history (accuracy/loss vs. epochs) for the DL model.
    *   Side-by-side confusion matrices for validation and test sets.
    *   Bar chart comparing performance metrics (Accuracy, Precision, Recall, F1-Score) on validation and test sets.

## Technologies Used

*   Python 3.x
*   TensorFlow (version 2.x, e.g., 2.10.0 as per notebook)
*   Keras (as part of TensorFlow)
*   Scikit-learn
*   NLTK
*   Pandas (e.g., 1.5.3)
*   NumPy (e.g., 1.23.5)
*   Matplotlib
*   Seaborn

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/IMDB-Sentiment-Analysis-ML-vs-DL.git
    cd IMDB-Sentiment-Analysis-ML-vs-DL
    ```
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    tensorflow>=2.9.0  # Or a specific version like 2.10.0
    scikit-learn
    nltk
    pandas
    numpy
    matplotlib
    seaborn
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```
4.  **NLTK Data:** The notebook automatically attempts to download the `stopwords` corpus from NLTK upon first run if not found. If you encounter issues, you can manually download it in a Python interpreter:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

## How to Run

1.  Ensure all dependencies are installed.
2.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```
3.  Open the `.ipynb` notebook file.
4.  **Model Selection:** Before running all cells, navigate to **Cell 8 (Model Selection and Hyperparameter Grid Definition)**. Modify the boolean flags (e.g., `IS_DEEP_LEARNING_MODEL`) and `MODEL_NAME` / `estimator_model` / `param_grid` assignments to choose the model you want to train and evaluate. For example, to run the CNN + BiLSTM model:
    ```python
    # --- Option 4: CNN + BiLSTM ---
    MODEL_NAME = "CNN + BiLSTM"
    IS_DEEP_LEARNING_MODEL = True
    # model and param_grid for Keras model are defined/handled later in specific Keras cells
    ```
    To run Logistic Regression:
    ```python
    # --- Option 1: Logistic Regression ---
    MODEL_NAME = "Logistic Regression"
    IS_DEEP_LEARNING_MODEL = False
    estimator_model = LogisticRegression(solver='saga', max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    param_grid = {'C': [0.1, 1, 10, 100]}
    ```
5.  Run the cells sequentially from top to bottom.

## Notebook Structure

The notebook is organized into the following main sections (cells):

*   **Cell 1:** Imports and Setup (MODIFIED)
*   **Cell 2:** Load Data (IMDB using Tensorflow Datasets)
*   **Cell 3:** Data Extract and Initial Inspection
*   **Cell 4:** Define Text Preprocessing Function
*   **Cell 5:** Apply Preprocessing to Datasets
*   **Cell 6:** Split Data into Training and Validation Sets
*   **Cell 7:** Feature Engineering (Conditional: TF-IDF or Skip for DL) (MODIFIED - Now Conditional)
*   **Cell 7b:** Tokenization and Padding (Runs ONLY for DL Model) (NEW)
*   **Cell 8:** Model Selection and Hyperparameter Grid Definition (MODIFIED)
*   **Cell 8b:** Define Keras Model Architecture (Runs ONLY for DL Model) (NEW)
*   **Cell 9:** Model Training (Conditional: GridSearchCV or Keras fit) (MODIFIED)
*   **Cell 10:** Evaluation on Validation Set (Conditional) (MODIFIED)
*   **Cell 11:** Evaluation on Test Set (Final Evaluation - Conditional) (MODIFIED)
*   **Cell 12:** Visualization - Combined Confusion Matrices
*   **Cell 13:** Visualization - Metrics Comparison Bar Chart

## Example Results

The notebook outputs detailed performance metrics and visualizations for the selected model. For the **CNN + BiLSTM model**, typical results (as seen in the provided sample output) are around:

*   **Validation Accuracy:** ~87-88%
*   **Test Accuracy:** ~85-86%

Confusion matrices and a bar chart comparing Accuracy, Precision, Recall, and F1-score for both validation and test sets are generated, providing a clear view of the model's performance.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details (you would create a `LICENSE` file with MIT license text).
