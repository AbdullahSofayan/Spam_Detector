import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
spam_df = pd.read_csv("spam.csv")

# Creating a new column 'spam' to identify category numerically (spam -> 1, ham -> 0)
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Splitting into test and train
x_train, x_test, y_train, y_test = train_test_split(spam_df.Message, spam_df.spam, test_size=0.25, random_state=42)

# Convert text to numerical data using CountVectorizer
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)

# Train model
model = MultinomialNB()
model.fit(x_train_count, y_train)

# Function to predict if a message is spam or ham
def predict_spam(message: str):
    message_count = cv.transform([message])  # Transform input message
    prediction = model.predict(message_count)[0]  # Return prediction (0 for ham, 1 for spam)
    return "Spam" if prediction == 1 else "Ham"

# Menu system
while True:
    print("\n==== Spam Detection Menu ====")
    print("1. Test a message")
    print("2. Test model accuracy")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        user_message = input("\nEnter the message: ")
        result = predict_spam(user_message)
        print(f"\nPrediction: {result}")
    
    elif choice == "2":
        x_test_count = cv.transform(x_test)
        accuracy = model.score(x_test_count, y_test)
        print(f"\nModel Accuracy: {accuracy:.2%}")
    
    elif choice == "3":
        print("\nExiting... Goodbye!")
        break
    
    else:
        print("\nInvalid choice! Please enter 1, 2, or 3.")
