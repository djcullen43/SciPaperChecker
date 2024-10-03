# AI-Generated Scientific Paper Detection
## Overview
This project implements a machine learning model to detect AI-generated content in scientific papers. Using Word2Vec embeddings and a Bidirectional LSTM neural network, the model classifies scientific papers based on their title, abstract, and introduction as either original or AI-generated. The goal of the project is to identify potential instances of plagiarism or AI-assisted paper generation.

## Features
### Text Preprocessing 
Combines and tokenizes the title, abstract, and introduction sections of scientific papers.
### Word2Vec Embeddings: 
Utilizes Word2Vec embeddings to create a dense vector representation of the text data.
### LSTM Neural Network:
A Bidirectional LSTM model is trained for binary classification (original vs AI-generated).

## How It Works
### Data Preparation

The input data consists of CSV files with columns for the title, abstract, introduction, and a binary label indicating whether the paper is AI-generated.
Text from the title, abstract, and introduction are combined and tokenized using a custom preprocessing function.
### Word2Vec Embeddings:

A Word2Vec model is trained on the tokenized text data to generate word embeddings, which capture semantic relationships between words.
### Text Vectorization

The tokenized text is converted into sequences and padded to ensure equal length input for the LSTM model.
### Model Training:

A Bidirectional LSTM model is trained to classify the papers as either AI-generated or original based on the vectorized text.
### Prediction:

After training, the model is used to predict the label for new test data. The results are saved in a CSV file.

