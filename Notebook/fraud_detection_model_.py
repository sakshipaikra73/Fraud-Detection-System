
import pandas as pd
import numpy as np
import re
import pickle
import joblib
import os
import math
from urllib.parse import urlparse
from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

# Define paths
DATA_PATH = "Notebook/your_real_data.csv"
MODEL_PATH = "Notebook/trained_models/best_model.pkl"

# --- Feature Engineering ---

def calculate_entropy(text):
    if not text:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(text.count(chr(x))) / len(text)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for url in X:
            features.append(self._extract(url))
        return pd.DataFrame(features)

    def _extract(self, url):
        features = {}
        
        # 1. URL Length
        features['url_length'] = len(url)
        
        # 2. Number of digits
        features['digit_count'] = sum(c.isdigit() for c in url)
        
        # 3. Special characters count
        special_chars = "!@#$%^&*()_+-=[]{};':\",.<>/?\\|"
        features['special_char_count'] = sum(c in special_chars for c in url)
        
        # 4. Entropy
        features['entropy'] = calculate_entropy(url)
        
        # 5. Suspicious keywords count (e.g., login, verify, update, account, banking)
        keywords = ['login', 'verify', 'update', 'account', 'banking', 'secure', 'confirm', 'signin']
        features['suspicious_keywords'] = sum(keyword in url.lower() for keyword in keywords)
        
        # 6. Presence of IP address
        # Regex for IPv4
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        features['has_ip'] = 1 if re.search(ip_pattern, url) else 0
        
        # 7. HTTPS usage
        features['is_https'] = 1 if url.lower().startswith('https://') else 0
        
        # Additional: Subdomain level depth
        parsed = urlparse(url)
        hostname = parsed.netloc
        features['subdomain_depth'] = hostname.count('.') if hostname else 0

        # Additional: Hostname length
        features['hostname_length'] = len(hostname) if hostname else 0
        
        # Additional: Path length
        features['path_length'] = len(parsed.path) if parsed.path else 0
        
        return features

# --- Pipeline Construction ---

def create_pipeline(model_type='rf'):
    # Feature extraction transformer
    feature_extractor = URLFeatureExtractor()
    
    # TF-IDF transformer on character level
    # We apply TF-IDF on the raw URL string
    tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=5000)
    
    # ColumnTransformer to combine manually extracted features and TF-IDF features
    # But wait, ColumnTransformer applies to columns of a DataFrame. 
    # Our input X is a Series of URLs.
    # To use ColumnTransformer, we need to treat X as a DataFrame or handle it carefully.
    # A cleaner approach for raw text input in sklearn pipeline is to branch:
    # 1. Pipeline(FeatureExtractor -> Dict/DataFrame)
    # 2. Pipeline(Tfidf)
    # using FeatureUnion. But FeatureUnion is for transformers that return arrays.
    
    # Let's keep it simple. We'll define a custom transformer that handles the branching internally or use a ColumnTransformer 
    # but we need to pass a DataFrame with 'url' column.
    
    # Simpler approach: 
    # We will assume input to pipeline is a Series of URLs. 
    # We need two transformers: one that takes the Series and generates numeric features,
    # another that takes the Series and generates TF-IDF.
    pass

# Correct approach using FeatureUnion equivalent or ColumnTransformer on a DataFrame
# We will assume the input to the pipeline's fit/predict is a pandas Series or DataFrame column 'url'.

from sklearn.pipeline import FeatureUnion

class SeriesSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key=None):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Assumes X is a DataFrame
        return X[self.key]

class FeatureExtractorWrapper(BaseEstimator, TransformerMixin):
    """Wrapper to make URLFeatureExtractor compatible with FeatureUnion/Pipeline expectations"""
    def __init__(self):
        self.extractor = URLFeatureExtractor()
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # X is a Series of URLs
        return self.extractor.transform(X)

def build_model_pipeline(classifier):
    
    # We process the 'url' column in two ways:
    # 1. Structural features
    # 2. TF-IDF features
    
    features = FeatureUnion([
        ('structural_features', FeatureExtractorWrapper()),
        ('tfidf_features', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=1000))
    ])
    
    pipeline = Pipeline([
        ('features', features),
        ('classifier', classifier)
    ])
    
    return pipeline

# --- Main Execution ---

def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Creating dummy dataset...")
        # Create dummy dataset
        data = {
            'url': [
                'https://www.google.com', 'http://phishing-site.com/login', 
                'https://www.amazon.com', 'http://192.168.1.1/update-info',
                'https://github.com', 'http://secure-banking-verify.com/account',
                'https://stackoverflow.com', 'http://free-iphone.net/claim',
                'https://cnn.com', 'http://paypal-security-check.com/signin',
                'https://microsoft.com', 'http://apple-id-reset.com',
                'https://wikipedia.org', 'http://netflix-payment-failed.com',
                'https://linkedin.com', 'http://facebook-login-verify.com',
                'https://youtube.com', 'http://urgent-message.com/verify'
            ] * 50, # 900 samples
            'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 50
        }
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        return df
    
    try:
        df = pd.read_csv(DATA_PATH)
        # Ensure 'url' and 'label' columns exist
        if 'url' not in df.columns or 'label' not in df.columns:
             # Try assuming first column is url, second is label if names don't match
             df.columns = ['url', 'label'] + list(df.columns[2:])
        return df.dropna()
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train_model():
    print("Loading data...")
    df = load_data()
    if df is None:
        return
    
    X = df['url']
    y = df['label']
    
    print(f"Data loaded. Shape: {df.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Models to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    # Add XGBoost if available
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    except ImportError:
        print("XGBoost not installed. Skipping.")
    
    best_model = None
    best_score = 0
    best_name = ""
    
    results = {}
    
    print("\nTraining and comparing models...")
    
    for name, clf in models.items():
        print(f"Testing {name}...")
        pipeline = build_model_pipeline(clf)
        
        # Cross-validation
        cv_scores = []
        skf = StratifiedKFold(n_splits=5)
        
        # We need to manually do CV loop if we want to be correct with Pipeline, 
        # but cross_val_score handles pipeline correctly.
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
        mean_cv_score = scores.mean()
        print(f"  {name} CV ROC-AUC: {mean_cv_score:.4f}")
        
        # Train on full training set for final eval
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  Test ROC-AUC: {roc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        
        results[name] = {'model': pipeline, 'score': roc}
        
        if roc > best_score:
            best_score = roc
            best_model = pipeline
            best_name = name
            
    print(f"\nBest Model: {best_name} with ROC-AUC: {best_score:.4f}")
    
    # Save best model
    if best_model:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"Best model saved to {MODEL_PATH}")

def predict_url(url, model=None):
    if model is None:
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            return {"error": f"Could not load model: {e}"}
            
    # Model expects an iterable of strings (Series or list)
    prediction_prob = model.predict_proba([url])[0][1]
    prediction_class = model.predict([url])[0]
    
    return {
        "url": url,
        "is_fraud": bool(prediction_class),
        "confidence": float(prediction_prob) * 100,
        "risk_level": "High" if prediction_prob > 0.7 else "Medium" if prediction_prob > 0.4 else "Low"
    }

def extract_features_standalone(url):
    """Wrapper to expose feature extraction logic without pipeline dependency if needed."""
    extractor = URLFeatureExtractor()
    return extractor._extract(url)

if __name__ == "__main__":
    train_model()
