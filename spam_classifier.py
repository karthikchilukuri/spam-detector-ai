import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# ----------------------------
# 1. LOAD DATASET
# ----------------------------
# Make sure spam.csv is in same folder

df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only useful columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ----------------------------
# 2. SPLIT DATA
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# ----------------------------
# 3. TF-IDF VECTORIZATION
# ----------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# 4. TRAIN MODEL
# ----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ----------------------------
# 5. EVALUATION
# ----------------------------
y_pred = model.predict(X_test_vec)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# 6. TEST WITH CUSTOM INPUT
# ----------------------------
while True:
    msg = input("\nEnter a message (or type 'exit'): ")
    
    if msg.lower() == 'exit':
        break

    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)[0]

    if prediction == 1:
        print("🚨 Spam Message")
    else:
        print("✅ Not Spam")