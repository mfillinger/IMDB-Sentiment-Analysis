import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#load dataset
df = pd.read_csv("IMDB Dataset.csv")

#convert labels to numbers
df['sentiment']= df['sentiment'].map({'positive': 1, 'negative': 0})

#split data
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

#tf-idf vectorization
vectorizer= TfidfVectorizer(max_features=5000)

X_train_tfidf= vectorizer.fit_transform(X_train)
X_test_tfidf= vectorizer.transform(X_test)

print("Data loaded and vectorized successfully")
print("Training samples:", X_train_tfidf.shape)
print("Testing samples:", X_test_tfidf.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

#make predictions
y_pred = model.predict(X_test_tfidf)

#evaluate
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

from sklearn.metrics import classification_report

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
