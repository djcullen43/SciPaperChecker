import pandas as pd
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model = load_model('ai_generated_paper_detection_model.keras')

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()  # Tokenization

# Load the test data
test_df = pd.read_csv('train.csv')

# Preprocess the test data
test_X = test_df[['title', 'abstract', 'introduction']]
test_X.loc[:, 'combined_text'] = test_X['title'] + ' ' + test_X['abstract'] + ' ' + test_X['introduction']
test_X['tokenized_text'] = test_X['combined_text'].apply(preprocess_text)

# Tokenization and padding
max_words = 10000
maxlen = 200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(test_X['combined_text'])
test_sequences = tokenizer.texts_to_sequences(test_X['combined_text'])
test_X_pad = pad_sequences(test_sequences, maxlen=maxlen)

# Make predictions on the test data
test_probabilities = model.predict(test_X_pad)
test_predictions = (test_probabilities > 0.5).astype('int32')

# Get actual labels
y_true = test_df['label']

# Calculate performance metrics
accuracy = accuracy_score(y_true, test_predictions)
precision = precision_score(y_true, test_predictions)
recall = recall_score(y_true, test_predictions)
f1 = f1_score(y_true, test_predictions)

# Print classification report
report = classification_report(y_true, test_predictions)
print(report)

# Save report to a file
with open('test_classification_report.txt', 'w') as file:
    file.write(report)

# Plot and save confusion matrix
conf_matrix = confusion_matrix(y_true, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-AI Generated', 'AI-Generated'], yticklabels=['Non-AI Generated', 'AI-Generated'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')  # Save confusion matrix as an image
plt.close()

# Save predictions to a CSV file
with open('test_predictions.csv', 'w') as file:
    file.write("ID,label\n")
    for i, pred in enumerate(test_predictions):
        file.write(f"{i},{pred[0]}\n")

print("Test predictions and statistics saved. Confusion matrix and performance metrics are saved as images.")
