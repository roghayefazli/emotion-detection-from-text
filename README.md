### Emotion Detection with CNN

This Jupyter Notebook, `emotion_detection.ipynb`, demonstrates a machine learning approach to **emotion detection from text**. The notebook implements a Convolutional Neural Network (CNN) to classify text data as either `positive` or `negative`.

#### Project Overview

The project follows a standard text classification pipeline:

1.  **Data Loading and Preprocessing:** The notebook is set up for a Google Colab environment and loads a dataset from a `data.zip` file. The data is organized into `neg` and `pos` folders. The script then performs several text cleaning steps, including tokenization, removing punctuation, and filtering out common English stop words.
2.  **Model Architecture:** The notebook defines a CNN model using the Keras API in TensorFlow. The model architecture includes:
      * An **`Embedding`** layer to represent words as dense vectors.
      * A **`Conv1D`** layer for feature extraction from the text sequence.
      * A **`MaxPool1D`** layer to downsample the feature maps.
      * A **`Flatten`** layer to prepare the data for the dense layers.
      * **`Dense`** layers for classification, with a final layer using a `sigmoid` activation function for binary classification.
3.  **Training and Evaluation:** The script splits the preprocessed data into training and testing sets, then trains the model on the training data. The tokenizer is saved as `tokenizer.h5`.

#### Requirements

To run this notebook, you will need the following libraries:

  * `os`
  * `nltk`
  * `random`
  * `tensorflow`

Additionally, you need to download the `punkt` and `stopwords` corpora from NLTK. This can be done by running the following command in a Python environment:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

#### How to Use

1.  **Environment:** Open this notebook in a Google Colab environment.
2.  **Data:** Ensure you have a `data.zip` file containing `neg` and `pos` folders for your negative and positive text data, respectively. This file should be available at the path `/content/data.zip`.
3.  **Execution:** Run all cells in the notebook sequentially to load the data, preprocess the text, build and train the CNN model, and save the tokenizer.
