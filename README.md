# Social Media NLP Pipeline

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)

A comprehensive Natural Language Processing (NLP) pipeline for analyzing sentiment and topics in social media data, specifically designed for the Twitter US Airline Sentiment dataset. This project combines text preprocessing, machine learning models for sentiment classification, and topic modeling to provide actionable insights from social media conversations.

## 🚀 Features

- **Text Preprocessing**: Advanced cleaning, tokenization, lemmatization, and stopword removal using NLTK and spaCy
- **Sentiment Analysis**: TF-IDF vectorization with Logistic Regression for multi-class sentiment classification (positive, negative, neutral)
- **Topic Modeling**: Latent Dirichlet Allocation (LDA) with interactive visualizations using pyLDAvis
- **Interactive Dashboard**: Streamlit-based web application for exploring results and visualizations
- **Containerized Deployment**: Docker and Docker Compose setup for easy deployment and reproducibility
- **Model Persistence**: Save and load trained models for inference and deployment

## 📊 Dataset

This project uses the [Twitter US Airline Sentiment Dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) from Kaggle, which contains:
- 14,640 tweets about major US airlines
- Sentiment labels: positive, negative, neutral
- Additional metadata: airline, tweet location, etc.

## 🛠️ Installation

### Prerequisites
- Docker and Docker Compose
- Git

### Quick Start with Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/social-media-nlp.git
   cd social-media-nlp
   ```

2. **Download the dataset**:
   - Visit [Kaggle Twitter Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
   - Download `Tweets.csv` and place it in the `data/` directory

3. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

4. **Access the application**:
   - Open your browser and navigate to `http://localhost:8501`

### Local Development Setup

1. **Clone and navigate**:
   ```bash
   git clone https://github.com/your-username/social-media-nlp.git
   cd social-media-nlp
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **Download dataset** (as above)

5. **Run preprocessing and training**:
   ```bash
   python src/preprocess.py
   python src/sentiment_model.py
   python src/topic_model.py
   ```

6. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## 📈 Usage

### Training Pipeline

The pipeline consists of three main steps:

1. **Preprocessing** (`src/preprocess.py`):
   - Cleans raw tweet text
   - Removes URLs, mentions, hashtags, and punctuation
   - Performs lemmatization and stopword removal

2. **Sentiment Model Training** (`src/sentiment_model.py`):
   - Trains a Logistic Regression classifier on TF-IDF features
   - Evaluates model performance with accuracy, precision, recall, and F1-score
   - Saves model artifacts and predictions

3. **Topic Modeling** (`src/topic_model.py`):
   - Applies LDA to discover latent topics in the corpus
   - Generates interactive visualizations
   - Saves topic-word distributions

### Web Dashboard

The Streamlit application provides:
- **Sentiment Metrics**: Model performance statistics
- **Topic Discovery**: Top words for each discovered topic
- **LDA Visualization**: Interactive topic exploration with pyLDAvis

## 🏗️ Project Structure

```
social-media-nlp/
├── data/
│   └── Tweets.csv                    # Raw dataset
├── output/
│   ├── preprocessed_data.csv         # Cleaned text data
│   ├── sentiment_metrics.json        # Model evaluation metrics
│   ├── sentiment_predictions.csv     # Test set predictions
│   ├── topics.json                   # Discovered topics
│   ├── lda_visualization.html        # Interactive topic visualization
│   ├── tfidf_vectorizer.pkl          # TF-IDF vectorizer
│   ├── sentiment_model.pkl           # Trained sentiment classifier
│   └── lda_model.pkl                 # Trained LDA model
├── src/
│   ├── __init__.py
│   ├── preprocess.py                 # Text preprocessing module
│   ├── sentiment_model.py            # Sentiment analysis module
│   └── topic_model.py                # Topic modeling module
├── app.py                            # Streamlit web application
├── Dockerfile                        # Docker image definition
├── docker-compose.yml                # Container orchestration
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

## 🔧 Configuration

### Model Parameters

- **Sentiment Model**:
  - TF-IDF max_features: 5000
  - Logistic Regression max_iter: 1000
  - Test size: 20% with stratification

- **Topic Model**:
  - Number of topics: 5
  - LDA max_df: 0.9, min_df: 10

### Environment Variables

The application uses default paths. For custom configurations, modify the scripts directly or set environment variables in Docker Compose.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Twitter US Airline Sentiment Dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) by CrowdFlower
- [spaCy](https://spacy.io/) for NLP processing
- [scikit-learn](https://scikit-learn.org/) for machine learning
- [Streamlit](https://streamlit.io/) for the web interface
- [pyLDAvis](https://github.com/bmabey/pyLDAvis) for topic visualization

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: Ensure you have the necessary permissions and comply with Twitter's Terms of Service when working with tweet data.