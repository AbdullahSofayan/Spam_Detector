import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import tkinter as tk
from tkinter import messagebox

# Load dataset
spam_df = pd.read_csv("spam.csv")

# Create 'spam' column (spam -> 1, ham -> 0)
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Split data
x_train, x_test, y_train, y_test = train_test_split(spam_df.Message, spam_df.spam, test_size=0.25, random_state=42)

# Vectorize text
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)

# Train model
model = MultinomialNB()
model.fit(x_train_count, y_train)

# Predict function
def predict_spam(message):
    message_count = cv.transform([message])
    prediction = model.predict(message_count)[0]
    return "Spam" if prediction == 1 else "Ham"

# Accuracy function
def show_accuracy():
    x_test_count = cv.transform(x_test)
    accuracy = model.score(x_test_count, y_test)
    messagebox.showinfo("Model Accuracy", f"Accuracy: {accuracy:.2%}")

# Predict button action
def on_predict():
    msg = entry.get("1.0", tk.END).strip()
    if msg:
        result = predict_spam(msg)
        result_label.config(text=f"Prediction: {result}")
    else:
        messagebox.showwarning("Input Error", "Please enter a message.")

# GUI setup
root = tk.Tk()
root.title("Spam Detector")

tk.Label(root, text="Enter your message:").pack(pady=5)
entry = tk.Text(root, height=4, width=50)
entry.pack()

tk.Button(root, text="Predict", command=on_predict).pack(pady=5)
result_label = tk.Label(root, text="Prediction: ", font=("Arial", 12, "bold"))
result_label.pack(pady=5)

tk.Button(root, text="Show Model Accuracy", command=show_accuracy).pack(pady=5)

tk.Button(root, text="Exit", command=root.quit).pack(pady=10)

root.mainloop()
