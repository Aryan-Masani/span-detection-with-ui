import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords # it will ignore common words like 'the', 'is', etc.
from nltk.stem import PorterStemmer # it will reduce words to their root form eg 'playing' to 'play' talking to talk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))# we dont want to use stop words in our text processing and it will be ignored

def preprocess_text(text):

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Stem and remove stop words
    return ' '.join(words)  # Join words back into a single string



df = pd.read_csv('spam.csv', encoding='latin-1' )[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

df['cleaned_text'] = df['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=3000)  # Limit to 3000 features
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}')
print(classification_report(y_test, y_pred))

def predict(email_text):
    cleaned_text = preprocess_text(email_text)              # text ko clean karna hai -> vectorize transform karna hai
    # Vectorize the cleaned text using the same vectorizer used for training

    vectorized_text = vectorizer.transform([cleaned_text])# phir model se prediction karna hai
    prediction = model.predict(vectorized_text)
    return 'spam' if prediction[0] == 1 else 'not spam'

email_text = "Congratulations! You've won a lottery of $1000. Click here to claim your prize."
result = predict(email_text)
print(f'The email is classified as: {result}')

# Example usage
femail_text = "Hello, I hope you are doing well. Let's catch up soon!"
fresult = predict(femail_text)
print(f'The email is classified as: {fresult}')