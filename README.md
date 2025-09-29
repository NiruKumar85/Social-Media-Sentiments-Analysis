# 📢 Social Media Sentiment Analysis

This project performs **Sentiment Analysis** on social media data to understand public opinions, emotions, and attitudes expressed in posts and comments.  
The goal is to classify text into categories such as **Positive, Negative, and Neutral** and provide insights into overall sentiment trends.  

---

## 📌 Project Overview
- Collect and preprocess **social media text data** (Twitter, Facebook, Instagram, etc.).  
- Clean data by removing **stopwords, punctuation, hashtags, mentions, and URLs**.  
- Apply **Natural Language Processing (NLP)** techniques for feature extraction:
  - Bag of Words (BoW)
  - TF-IDF
  - Word Embeddings (Word2Vec, GloVe, FastText)  
- Train sentiment classification models:
  - Logistic Regression
  - Naïve Bayes
  - Support Vector Machines (SVM)
  - Deep Learning models (LSTMs, Transformers like BERT)  
- Visualize sentiment distribution and trends over time.  

---

## 📂 Project Structure



Social-Media-Sentiments-Analysis/
│
├── data/ # Dataset(s)
├── notebooks/ # Jupyter notebooks for EDA & modeling
├── src/ # Source code for preprocessing & model building
├── results/ # Sentiment analysis results & visualizations
├── requirements.txt # Python dependencies
└── README.md # Project documentation







---

## 🚀 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/NiruKumar85/Social-Media-Sentiments-Analysis.git
   cd Social-Media-Sentiments-Analysis





2. Create & activate virtual environment:
   python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows


3. Install dependencies:
   pip install -r requirements.txt


🛠️ Technologies Used

Python

NLTK, SpaCy – Text preprocessing & tokenization

Scikit-learn – Machine Learning models (Logistic Regression, Naïve Bayes, SVM)

TensorFlow / PyTorch – Deep Learning models (LSTMs, RNNs, Transformers)

Transformers (Hugging Face) – BERT, RoBERTa for advanced sentiment classification

Matplotlib, Seaborn, Plotly – Data visualization and interactive charts






📈 Sample Visualizations

Word Clouds for Positive & Negative tweets

Bar Charts showing sentiment distribution

Line Plots showing sentiment variation over time

Heatmaps for correlation between features and sentiment



📊 Results

Sentiment classification into Positive, Negative, and Neutral categories.

Achieved higher accuracy using transformer-based models compared to traditional ML approaches.

Extracted insights into public opinion trends from social media posts.

