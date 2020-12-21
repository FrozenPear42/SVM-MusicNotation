import os
import os.path
import glob
import json
import random
import cv2
import sys
import mahotas
import numpy as np
import sklearn.preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
import argparse


def train(dataset_path, model_path):
    classes = {}

    for meta in glob.glob(dataset_path + "/*.json"):
        with open(meta, "r") as file:
            data = json.load(file)
            classes = {**classes, **data}

    dataset = [(os.path.join(dataset_path, ex[0]), ex[1]) for ex in list(classes.items())]

    dataset_x = [x[0] for x in dataset]
    dataset_y = [x[1] for x in dataset]

    x_train, x_test, y_train, y_test = train_test_split(
        dataset_x, dataset_y, test_size=0.33, random_state=4, stratify=dataset_y)

    input_size = (0, 4116)

    xs = np.empty(input_size, float)
    ys = []
    train_count = len(x_train)
    for idx, ex in enumerate(x_train):
        if idx % 500 == 0:
            print(f"fprgress: {idx}/{train_count}")

        try:
            features = process_input(ex)
            xs = np.append(xs, [features], axis=0)
            ys.append(y_train[idx])
        except Exception as e:
            print(e)

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler.fit(xs)

    xs_scaled = scaler.transform(xs)

    classifier = SVC(verbose=True)
    classifier.fit(xs_scaled, ys)

    save_model(model_path, scaler, classifier)

    xs_test = np.empty(input_size, float)
    ys_test = y_test
    for ex in x_test:
        try:
            features = process_input(ex)
            xs_test = np.append(xs_test, [features], axis=0)
        except Exception as e:
            print(e)

    score = classifier.score(xs_test, ys_test)
    print(f"Score: {score}")


def predict(model_path, args):
    scaler, classifier = load_model(model_path)

    if args.interactive:
        while True:
            print("enter path: ")
            inp = sys.stdin.readline()
            if inp == '':
                break
            inp = inp.strip()
            print(inp)
            try:
                features = process_input(inp)
                features_scaled = scaler.transform([features])
                print(features_scaled)
                prediction = classifier.predict(features_scaled)
                print(prediction)
            except Exception as e:
                print(f"error {e}")
    else:
        features = process_input(args.image)
        features_scaled = scaler.transform([features])
        prediction = classifier.predict(features_scaled)
        print(f"predicted class: {prediction}")


def process_input(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"image is none: {image_path}")
        raise Exception(image_path)
    features = extract_image_features(img)
    return features


def extract_image_features(image):
    hu_moments = extract_hu_moments(image)
    haralick = extract_haralick_features(image)
    histogram = extract_histogram(image)
    features = np.hstack([hu_moments, haralick, histogram])
    return features


def extract_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def extract_haralick_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick


def extract_histogram(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([image], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None)
    hist = hist.flatten()
    return hist


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("command", help="subcommand to run")
    args = parser.parse_args(sys.argv[1:2])
    if args.command == "train":
        pass
    elif args.command == "predict":
        sub_parser = argparse.ArgumentParser()
        sub_parser.add_argument("--interactive")
        sub_parser.add_argument("--image")
        sub_args = sub_parser.parse_args((sys.argv[2:]))
        return args.command, sub_args
    else:
        print('Unrecognized command')
        parser.print_help()
        exit(1)

    return args.command, None


def load_model(file_path):
    with open(file_path, "rb") as f:
        classifier_state = pickle.load(f)
        return classifier_state["scaler"], classifier_state["classifier"]


def save_model(file_path, scaler, classifier):
    classifier_state = {
        'scaler': scaler,
        'classifier': classifier,
    }
    with open(file_path, "wb") as f:
        pickle.dump(classifier_state, f)


def main():
    command, args = parse_arguments()
    if command == "train":
        train("./dataset/regular", "./model.pkl")
    elif command == "predict":
        predict("./model.pkl", args)


if __name__ == "__main__":
    main()
