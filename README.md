# Disaster Tweet Classification: An NLP Approach

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
This project aims to classify tweets into disaster-related and non-disaster-related categories using Natural Language Processing (NLP) techniques and machine learning models. The primary goal is to build a robust model that can automatically identify disaster-related content on Twitter, which could be useful for real-time disaster response and management.

## Dataset
The dataset used in this project is sourced from the [Kaggle competition "Natural Language Processing with Disaster Tweets"](https://www.kaggle.com/c/nlp-getting-started). It consists of labeled tweets indicating whether they are related to real disasters or not.

- **Training Data:** `train.csv` (7,613 tweets)
- **Test Data:** `test.csv` (3,263 tweets)
- **Columns:**
  - `id`: Unique identifier for the tweet.
  - `text`: The text of the tweet.
  - `location`: Location the tweet was sent from (may be blank).
  - `keyword`: A particular keyword from the tweet (may be blank).
  - `target`: Indicates if the tweet is about a real disaster (1) or not (0).

## Installation
To run this project, clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/yourusername/disaster-tweet-classification.git
cd disaster-tweet-classification
pip install -r requirements.txt
```

## Project Structure
```bash
disaster-tweet-classification/
├── data/
│   ├── train.csv
│   ├── test.csv
├── notebooks/
│   ├── eda.ipynb          # Exploratory Data Analysis
│   ├── modeling.ipynb     # Model development and evaluation
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── inference.py
├── README.md
├── requirements.txt
```

## Modeling Approach
The project involves several advanced NLP techniques and models to classify disaster tweets:

1. **Data Preprocessing:** 
   - Text data was cleaned by removing stopwords, special characters, and performing lemmatization.
   - Tokenization was applied to convert the text into sequences of tokens, which were then padded to ensure uniform input length.

2. **Feature Engineering:** 
   - Word embeddings were generated using pre-trained models like GloVe to represent words in a dense vector space.
   - Additionally, embeddings from DistilBERT, a smaller and faster version of BERT, were used to capture contextual information from the tweets.

3. **Model Selection:**
   - **LSTM (Long Short-Term Memory):** A deep learning model that is well-suited for sequence prediction tasks. The LSTM model was trained to capture temporal dependencies and context within the tweet sequences.
   - **RNN + DistilBERT:** A combination of a Recurrent Neural Network (RNN) architecture with DistilBERT embeddings. This model leverages the power of RNNs to process sequences and the contextual understanding provided by DistilBERT, enhancing the classification accuracy.

4. **Evaluation:** 
   - Models were evaluated using a range of metrics, including accuracy, F1-score, precision, and recall.
   - Cross-validation was used to ensure the robustness of the models, and hyperparameter tuning was conducted to optimize performance.

---

This revised section accurately reflects the advanced modeling techniques you employed, showcasing the use of LSTM, RNN, and DistilBERT in your approach.


## Results
The best model achieved an accuracy of **X%** and an F1-score of **Y** on the test dataset. The model shows strong potential for real-time disaster tweet classification, though further improvements can be made.

## Usage
To classify new tweets, use the `inference.py` script:

```bash
python src/inference.py --input "This is a sample tweet text."
```

The script will output whether the tweet is related to a disaster or not.

## Future Work
- **Model Improvement:** Explore advanced models such as BERT for better performance.
- **Real-Time Application:** Integrate the model into a real-time tweet monitoring system.
- **Multilingual Support:** Extend the model to handle tweets in multiple languages.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- Various open-source libraries like Scikit-learn, Pandas, and NLTK that made this project possible.
