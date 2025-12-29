import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from features import clean_text, get_extra_features

print("Loading data and extracting features...")
df = pd.read_json("data/raw/problems_data.jsonl", lines=True) 
df['text'] = (df['title'] + " " + df['description'] + " " + 
              df['input_description'] + " " + df['output_description']).apply(clean_text)

# Using ngrams (1,3) to capture complex phrases like "shortest path algorithm"
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 3), stop_words='english')
X_tfidf = tfidf.fit_transform(df['text']).toarray()
X_manual = np.array([get_extra_features(t) for t in df['text']])
X_final = np.hstack([X_tfidf, X_manual])

# Split for validation
X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X_final, df['problem_class'], df['problem_score'], test_size=0.2, random_state=42
)

print("Training ExtraTrees models for maximum accuracy...")
# ExtraTrees is often superior to Random Forest for noisy text features
clf = ExtraTreesClassifier(n_estimators=300, class_weight='balanced_subsample', n_jobs=-1, random_state=42)
reg = ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=42)

clf.fit(X_train, y_class_train)
reg.fit(X_train, y_score_train)

# Evaluation
y_pred = clf.predict(X_test)
print(f"\nFinal Accuracy: {accuracy_score(y_class_test, y_pred) * 100:.2f}%")
print(classification_report(y_class_test, y_pred))

# Save the final high-accuracy models
clf.fit(X_final, df['problem_class'])
reg.fit(X_final, df['problem_score'])
joblib.dump(tfidf, "data/processed/tfidf.pkl")
joblib.dump(clf, "data/processed/classifier.pkl")
joblib.dump(reg, "data/processed/regressor.pkl")
print("SUCCESS: High-accuracy models saved!")
