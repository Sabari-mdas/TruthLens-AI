"""
TruthLens AI — Fake News Model Training Script
================================================
This script trains a Naive Bayes classifier for fake news detection.

Usage:
  Option A: Place Kaggle's Fake.csv and True.csv in the data/ folder, then run:
      python train_model.py

  Option B: If no CSV files are found, the script generates synthetic demo data
            and trains a model on that instead.

Outputs:
  - model.pkl       (trained Multinomial Naive Bayes classifier)
  - vectorizer.pkl  (fitted TF-IDF vectorizer)
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib


def load_kaggle_data():
    """Try to load Fake.csv and True.csv from the data/ directory."""
    fake_path = os.path.join("data", "Fake.csv")
    true_path = os.path.join("data", "True.csv")

    if os.path.exists(fake_path) and os.path.exists(true_path):
        print("[INFO] Found Kaggle datasets. Loading...")
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)

        fake_df["label"] = 1  # 1 = Fake
        true_df["label"] = 0  # 0 = Real

        # Use the 'text' column; fall back to 'title' if 'text' is missing
        text_col = "text" if "text" in fake_df.columns else "title"

        df = pd.concat([fake_df[[text_col, "label"]], true_df[[text_col, "label"]]], ignore_index=True)
        df.rename(columns={text_col: "text"}, inplace=True)
        df.dropna(subset=["text"], inplace=True)
        return df
    return None


def generate_synthetic_data():
    """Generate synthetic demo data for training when real data is unavailable."""
    print("[INFO] No Kaggle data found. Generating synthetic demo data...")

    fake_samples = [
        "BREAKING: Scientists discover miracle cure that heals everything instantly!",
        "SHOCKING! Government secretly implants chips in citizens through vaccines.",
        "URGENT: Drinking bleach proven to cure all diseases by secret research lab.",
        "Unbelievable! Man finds treasure worth billions hidden under his house.",
        "Breaking news: Aliens have landed and the government is hiding the truth!",
        "SHOCKING revelation: 5G towers are actually mind control devices!",
        "Miracle cure found in common household item — doctors don't want you to know!",
        "URGENT: The moon landing was faked and NASA finally admits it!",
        "Secret study proves that eating chocolate makes you lose weight instantly.",
        "Breaking! Famous actor reveals the government is run by lizard people.",
        "Unbelievable discovery: Water has memory and can cure cancer instantly!",
        "SHOCKING: Social media is reading your thoughts through your phone camera.",
        "Miracle herb cures diabetes, cancer, and heart disease overnight!",
        "BREAKING: Secret documents prove the earth is actually flat!",
        "URGENT WARNING: WiFi signals cause brain damage according to leaked study.",
        "Scientists baffled as man claims to live without food for 50 years!",
        "SHOCKING truth about vaccines that mainstream media refuses to report!",
        "Breaking exclusive: Celebrity endorses dangerous miracle weight loss pill.",
        "Unbelievable: Ancient remedy cures blindness instantly — suppressed by Big Pharma!",
        "URGENT: New world order plan leaked — everything you know is a lie!",
        "Miracle water from secret spring cures all known diseases!",
        "BREAKING: Time traveler from 2050 warns of impending catastrophe!",
        "SHOCKING: Popular food brand caught putting addictive chemicals in products!",
        "Secret research proves smartphones are making humans stupider!",
        "Unbelievable conspiracy: Birds aren't real — they're government drones!",
    ]

    real_samples = [
        "The Federal Reserve maintained interest rates at the current level after their quarterly meeting.",
        "Researchers at MIT published a peer-reviewed study on renewable energy efficiency improvements.",
        "The World Health Organization released updated guidelines for pandemic preparedness.",
        "A new study published in Nature examines the effects of climate change on coral reefs.",
        "The European Parliament voted on new data privacy regulations affecting tech companies.",
        "Scientists at CERN reported new measurements consistent with the Standard Model of physics.",
        "The International Monetary Fund released its quarterly global economic outlook report.",
        "A clinical trial for a new Alzheimer's treatment showed promising results in Phase 3.",
        "The United Nations held a summit on sustainable development goals progress.",
        "NASA's Mars rover discovered mineral deposits suggesting past water activity.",
        "The Centers for Disease Control released annual flu vaccination recommendations.",
        "A long-term study found that regular exercise reduces cardiovascular disease risk by 30%.",
        "The Supreme Court heard arguments on a landmark digital privacy case.",
        "Economists at Oxford University published research on post-pandemic recovery trends.",
        "The National Weather Service issued updated hurricane season forecasts.",
        "Peer-reviewed research in The Lancet confirmed vaccine efficacy rates above 90%.",
        "The government announced new infrastructure spending plans for rural broadband access.",
        "Agricultural scientists develop drought-resistant crop varieties through gene editing.",
        "International trade negotiations continue between major economic blocs.",
        "Public health officials recommend updated guidelines for childhood nutrition.",
        "New archaeological discoveries in Egypt reveal previously unknown ancient structures.",
        "The space agency successfully tested its next-generation rocket propulsion system.",
        "Financial regulators announce new rules for cryptocurrency trading platforms.",
        "University researchers develop more efficient solar cell technology using perovskites.",
        "Environmental agency reports improvement in air quality in major metropolitan areas.",
    ]

    texts = fake_samples + real_samples
    labels = [1] * len(fake_samples) + [0] * len(real_samples)

    df = pd.DataFrame({"text": texts, "label": labels})
    return df


def train():
    """Train the fake news detection model."""
    # Try loading real data first, fall back to synthetic
    df = load_kaggle_data()
    if df is None:
        df = generate_synthetic_data()

    print(f"[INFO] Dataset size: {len(df)} samples")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts().to_string()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Naive Bayes classifier
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[RESULTS] Accuracy: {accuracy:.4f}")
    print(f"\n[RESULTS] Classification Report:\n{classification_report(y_test, y_pred, target_names=['Real', 'Fake'])}")

    # Save model and vectorizer
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("\n[INFO] Saved model.pkl and vectorizer.pkl")


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    train()
