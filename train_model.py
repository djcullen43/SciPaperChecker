import pandas as pd
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

# Load the data from CSV
df = pd.read_csv('train.csv')

# Set label as target
X = df[['title', 'abstract', 'introduction']]
y = df['label']

# Combine text data
X.loc[:, 'combined_text'] = X['title'] + ' ' + X['abstract'] + ' ' + X['introduction']

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()  # Tokenization

X['tokenized_text'] = X['combined_text'].apply(preprocess_text)

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=X['tokenized_text'], vector_size=75, window=15, min_count=2 , workers=4)

# Generating embeddings for each word in the text data
word_embeddings = dict(zip(word2vec_model.wv.index_to_key, word2vec_model.wv.vectors))

# Tokenization and padding
max_words = 10000
maxlen = 200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X['combined_text'])
sequences = tokenizer.texts_to_sequences(X['combined_text'])
X_pad = pad_sequences(sequences, maxlen=maxlen)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Create an embedding matrix using Word2Vec
word_index = tokenizer.word_index
num_words = min(max_words, len(word_index) + 1)
embedding_dim = 75

embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= max_words:
        continue
    if word in word_embeddings:
        embedding_vector = word_embeddings[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# LSTM layer initialization
model = Sequential()
model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Save the model
model.save('ai_generated_paper_detection_model.keras')  # Save the model in Keras format

print("Model trained and saved.")
