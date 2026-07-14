import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv(
    
    "datasets/train_data.txt",
    sep=":::",
    engine="python",
    names=["ID", "Title", "Genre", "Plot"]
   
)
print(train.head())
print(train.columns)
print(train.shape)
print(train.head())
test = pd.read_csv(
    "datasets/test_data.txt",
    sep=":::",
    engine="python",
    names=["ID", "Title", "Plot"]
)


test_labels = pd.read_csv(
    "datasets/test_data_solution.txt",
    sep=":::",
    engine="python",
    names=["ID", "Title", "Genre", "Plot"]
)

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

X_train = tfidf.fit_transform(train["Plot"].astype(str))
X_test = tfidf.transform(test["Plot"].astype(str))


model = LogisticRegression(max_iter=1000)

model.fit(X_train, train["Genre"])

predictions = model.predict(X_test)

accuracy = accuracy_score(test_labels["Genre"], predictions)

print("Accuracy :", accuracy)


joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")

print("✅ model.pkl saved")
print("✅ tfidf.pkl saved")