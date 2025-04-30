import pandas as pd  # For data handling
from sklearn.model_selection import train_test_split  # For splitting data into training/testing sets
from sklearn.feature_extraction.text import CountVectorizer  # To convert text to numerical format
from sklearn.naive_bayes import MultinomialNB  # The classifier used for spam detection
import tkinter as tk  # GUI library
from tkinter import messagebox  # For showing pop-up messages

# Load dataset from CSV file
spam_df = pd.read_csv("spam.csv")

# Create 'spam' column where 'spam' -> 1 and 'ham' -> 0
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Split the dataset into training and test sets (75% train, 25% test)
x_train, x_test, y_train, y_test = train_test_split(spam_df.Message, spam_df.spam, test_size=0.25, random_state=42)

# Initialize CountVectorizer to convert text messages to numeric form
cv = CountVectorizer()

# Fit the vectorizer on training data and transform it
x_train_count = cv.fit_transform(x_train.values)




print(cv.vocabulary_)