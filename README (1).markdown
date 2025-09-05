# Fake News Classifier

This project implements a fake news classifier using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) units, built with TensorFlow/Keras. The model classifies news articles as fake (1) or real (0) based on their titles.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Usage Notes](#usage-notes)
- [Contributing](#contributing)
- [License](#license)

## Overview
The notebook processes a dataset of news articles, preprocesses the text using techniques like tokenization, stemming, and one-hot encoding, and trains an LSTM-based model to classify articles as fake or real. Two models are implemented: one without dropout and one with dropout layers to prevent overfitting. The model achieves high accuracy on the test set (~91.5% with dropout).

## Dataset
The model expects a CSV file (e.g., `train.csv`) with the following columns:
- `id`: Unique identifier for each article.
- `title`: Article title (used for classification).
- `author`: Author of the article (not used in this model).
- `text`: Full article text (not used in this model).
- `label`: Binary label (1 for fake, 0 for real).

**Note**: The dataset is not included in this repository. You can obtain it from sources like Kaggle (e.g., [Fake News Dataset](https://www.kaggle.com/c/fake-news/data)). Place the CSV in the appropriate directory (e.g., `../input/train/train.csv`) or modify the file path in the notebook.

## Requirements
- Python 3.6 or higher
- Libraries:
  - pandas
  - tensorflow
  - scikit-learn
  - nltk
  - numpy

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-classifier.git
   cd fake-news-classifier
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install pandas tensorflow scikit-learn nltk numpy
   ```
4. Download NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## How to Run
1. Ensure the dataset CSV is in the correct path or update the path in the notebook (`pd.read_csv('../input/train/train.csv')`).
2. Open the notebook:
   ```bash
   jupyter notebook fake-news-classifier-using-rnn-lstm.ipynb
   ```
3. Run all cells in Jupyter Notebook to:
   - Load and preprocess the data.
   - Convert titles to one-hot representations and pad sequences.
   - Train the LSTM models (with and without dropout).
   - Evaluate performance using a confusion matrix and accuracy score.
4. Results will be displayed in the notebook, including training/validation accuracy and loss.

## Model Architecture
The notebook implements two LSTM models:
1. **Basic Model**:
   - Embedding layer (5000 vocabulary size, 40-dimensional vectors, input length 20).
   - LSTM layer (100 units).
   - Dense layer with sigmoid activation for binary classification.
   - Compiled with binary cross-entropy loss and Adam optimizer.
2. **Model with Dropout**:
   - Same as above but adds dropout layers (0.3 rate) after the embedding and LSTM layers to reduce overfitting.

Training is performed over 10 epochs with a batch size of 64. The dataset is split into 67% training and 33% testing.

## Performance
- **Basic Model**: Achieves ~99.8% training accuracy but shows signs of overfitting (validation accuracy ~91.2%, with increasing validation loss).
- **Dropout Model**: Slightly lower training accuracy (~99.4%) but better generalization (validation accuracy ~91.5%).
- **Confusion Matrix** (Dropout Model):
  ```
  [[3090,  329],
   [ 180, 2436]]
  ```
- **Test Accuracy**: 91.57%

## Usage Notes
- The model uses only the `title` column for classification. You can extend it to include the `text` column for better accuracy.
- The dataset must be preprocessed to remove missing values (handled in the notebook with `data.dropna()`).
- Adjust `voc_size` (5000) or `sent_length` (20) in the notebook if working with a different dataset.
- If the dataset is large, ensure sufficient computational resources (GPU recommended for faster training).
- The notebook includes outputs from a previous run; clear outputs before running (`Kernel > Restart & Clear Output`) to avoid conflicts.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make changes and commit (`git commit -m "Add feature"`).
4. Push to your fork (`git push origin feature-branch`).
5. Create a pull request.

Suggestions for improvement:
- Add Convolutional Neural Network (CNN) layers to enhance feature extraction.
- Incorporate the `text` column for richer data.
- Experiment with hyperparameters (e.g., embedding size, LSTM units).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.