import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Sample text for sentiment analysis
text = [
      "I love this product!", 
      "This is the worst experience ever.", 
      "Absolutely fantastic service.", 
      "I am not happy with the quality.",
      "This is amazing! I will recommend it to everyone.",
      "This is bad",
      "Very disappointing experience. I will not buy again.",
      "I am extremely satisfied with my purchase."
      ]

labels = [1, 0, 1, 0, 1, 0, 0, 1]  # 1 for positive sentiment, 0 for negative sentiment

# Processing Function to preprocess the text
def preprocess(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')  # Keep 'not' for sentiment analysis
    words= [w for w in tokens if w not in stop_words] # Remove stopwords

    lemmatizer = WordNetLemmatizer()

    words = [lemmatizer.lemmatize(w) for w in words]

    return ' '.join(words)

cleaned_text = [preprocess(t) for t in text]

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cleaned_text)

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test the model

test_sentences = input("Enter your sentence: ")
test_cleaned = preprocess(test_sentences)
test_vec = vectorizer.transform([test_cleaned])

# Predict the sentiment
prediction = model.predict(test_vec)

print("sentence:", test_sentences)
print("Prediction:", "Positive" if prediction[0] == 1 else "Negative")
