import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# 1. LOAD DATA
# ----------------------------
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ----------------------------
# 2. SPLIT DATA
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# ----------------------------
# 3. TF-IDF VECTORIZER
# ----------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),      # unigrams + bigrams
    stop_words='english',
    max_features=5000       # limit features (better performance)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# 4. MODEL
# ----------------------------
model = LogisticRegression(max_iter=1000)   # avoid convergence warning
model.fit(X_train_vec, y_train)

# ----------------------------
# 5. EVALUATION
# ----------------------------
y_pred = model.predict(X_test_vec)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# 6. SAVE MODEL
# ----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n💾 Model and vectorizer saved successfully!")