import re
import nltk
import string
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import pandas as pd
import pprint
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import csv
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
import cv2
import numpy as np
import pytesseract
from PIL import Image
import glob
from PIL import Image
import pytesseract
from process_image import ChColor

ch = ChColor()

import os

folder = "test"
new_folder = "processed_images"


def process_n_get_text():
    """

    :return:
    """
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        ch.load_image(file_path, the_file, new_folder)
    img_data = {}
    for the_file in os.listdir(new_folder):
        file_path = os.path.join(new_folder, the_file)
        data = pytesseract.image_to_string(Image.open(file_path))
        question = clean(data)
        predicted = predict([question])

        img_data.update({the_file:{"question":question,"subject": predicted}})
        # os.unlink(file_path)
    return img_data


def train():
    """

    :return:
    """
    chem_data = pd.read_csv("chem.csv")
    maths_data = pd.read_csv('maths.csv')

    frames = [chem_data[:8515], maths_data]
    final_df = pd.concat(frames)

    df = final_df.sample(frac=1).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(df["Questions"], df["class"],
                                                        train_size=0.80, random_state=42)

    pipeline = Pipeline([('vectorizer', CountVectorizer()),
                         ('SGD', SGDClassifier())
                         ])

    model = pipeline.fit(X_train, y_train)
    with open("sub_clsfr.pickle", "wb") as sub_clsfr_model:
        joblib.dump(model, sub_clsfr_model)

    predicted = model.predict(X_test)

    stats = classification_report(y_test, predicted)
    print("Stats", stats)

    matrix = confusion_matrix(y_test, predicted)
    print("Confusion matrix", matrix)

    accuracy_report = accuracy_score(y_test, predicted)
    print("accuracy_report", accuracy_report)

    predicted = cross_val_predict(pipeline, X_train, y_train, cv=10)
    cross_val_score = metrics.accuracy_score(y_train, predicted)
    print("cross_val_score", cross_val_score)


def predict(questions):
    """

    :param questions:
    :return:
    """
    clf_model = joblib.load('sub_clsfr.pickle')
    predictions = clf_model.predict(questions)
    return predictions


def get_string(img_path):
    # Read image with opencv
    img = cv2.imread(img_path)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    #  Apply threshold to get image with only black and white
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img, lang='eng')

    return result


def clean(da):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in da if ch not in exclude)

    word_tokens = word_tokenize(s)

    stp = ' '.join(lemmatizer.lemmatize(w.lower()) for w in s.split() if w.lower() not in stop_words)

    pp = re.sub("^\d+\s|\s\d+\s|\s\d+$|\d+", " ", stp).replace("nota", "").replace("select", "")

    cc = re.sub(r"\b[a-z]\b", "", pp)

    return cc


def get_image_text(image_path):
    """

    :param image_path:
    :return:
    """

    img_data = []
    files = glob.glob(image_path + "*.jpg")
    for img in files:
        img_name = img.replace(image_path, "")
        question = get_string(img)
        question = clean(question)
        subject_predicted = predict([question])
        img_data.append([img_name, question, subject_predicted[0]])
    return img_data


if __name__ == "__main__":

    results = process_n_get_text()

    print(results)
    with open("test_image.csv", "w") as we:
        writer = csv.writer(we)
        writer.writerows(results)
