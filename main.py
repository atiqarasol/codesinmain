import pickle
import pandas as pd

# load csv file

df = pd.read_csv("heart.csv")

# selection dependant and independant
# Spliting the data
from sklearn.model_selection import train_test_split

X = df.drop("output", axis=1)
y = df["output"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## Build a model (Random forest classifier)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train);
## Evaluating the model

# intentiate the model

classifier = RandomForestClassifier()

# fit the model

classifier.fit(X_train, y_train)

# make pickle file for the model

model = pickle.dump(classifier, open("model.pkl", "wb"))

#   Training part is complete


from flask import Flask, request, jsonify
import numpy as np
app = Flask(__name__)

@app.route("/")
def home():
    return ("Hello world")
model = pickle.load(open("model.pkl", "rb"))



@app.route("/predict", methods=["POST"])
def predict():

    age = request.form.get('age')
    sex = request.form.get('sex')
    cp = request.form.get('cp')
    trtbps =request.form.get('trtbps')
    chol = request.form.get('chol')
    fbs = request.form.get('fbs')
    exng = request.form.get('exng')
    oldpeak = request.form.get('oldpeak')
    slp = request.form.get('slp')
    caa = request.form.get('caa')
    thall = request.form.get('thall')
    input_query = np.array([[age, sex, cp, trtbps, chol, fbs,
                             exng, oldpeak, slp, caa,thall]])

    result = model.predict(input_query)[0]

    return jsonify({'Heart Attack Risk': str(result)})
    if __name__ == '__main__':
        app.run(debug=True)

    print(input_query)


    result = model.predict(sc.transform(input_query))
    print(result)
    return jsonify({'placement': str(result)})


if __name__ == '__main__':
    app.run(debug=True)

