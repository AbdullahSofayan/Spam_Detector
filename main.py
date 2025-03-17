import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


docs = pd.read_table('sms.txt',header = None, names=['Class', 'sms'])
# print(docs.head())
# print(len(docs))
# ham_spam = docs.Class.value_counts()
# print(ham_spam)

# mapping lables to 0 and 1
docs['lable'] = docs.Class.map({'ham':0, 'spam':1})
# print(docs.head())

# now we drop the column 'Class'
docs = docs.drop('Class', axis=1)
# print(docs.head())

# convert to x and y, SMS --> x and lable --> y
x = docs.sms
y = docs.lable
# print(x.shape)
# print(y.shape)

#splitting into test and train
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
# print(x_train.head())
# print(y_train.head())

# vectorizing the sentences; removing stop words
vect = CountVectorizer(stop_words='english')

vect.fit(x_train)
# print(vect.vocabulary_)

# transforming the train and test datasets
x_train_transformed = vect.transform(x_train)
x_test_transformed = vect.transform(x_test)