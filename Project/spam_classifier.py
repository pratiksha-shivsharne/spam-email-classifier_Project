import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'spam': 1, 'ham': 0})

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data ready! Training samples:", X_train.shape[0])

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=['Ham', 'Spam']
))

# Build confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Draw heatmap
plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=['Ham', 'Spam'],
    yticklabels=['Ham', 'Spam'],
    cmap='Greens'
)

plt.title('Spam Classifier - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.savefig('confusion_matrix.png')
plt.show()

print("Confusion matrix saved as confusion_matrix.png")

# Top spam words
spam_msgs = df[df['label'] == 1]['message'].str.cat(sep=' ')

from collections import Counter

words = spam_msgs.lower().split()
top_words = Counter(words).most_common(10)

labels, counts = zip(*top_words)

plt.figure(figsize=(8,5))
plt.bar(labels, counts)

plt.title('Top 10 Words in Spam Messages')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('spam_words.png')
plt.show()

print("Spam words chart saved!")

# Predict a new message
def predict_email(text):
    transformed = vectorizer.transform([text])
    result = model.predict(transformed)[0]
    return "SPAM" if result == 1 else "HAM (Not Spam)"

print("\nCustom Predictions:")
print(predict_email("Congratulations! You won a FREE prize. Call now"))
print(predict_email("Hi, meeting at 3pm tomorrow?"))