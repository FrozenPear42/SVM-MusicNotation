import os
import os.path
import glob
import json
import cv2
import sys
import mahotas
import numpy as np
import sklearn.preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
import pickle
import argparse


def prepare_dataset(dataset_path, output_path):
    classes = {}
    input_size = (0, 4116)

    for meta in glob.glob(dataset_path + "/*.json"):
        with open(meta, "r") as file:
            data = json.load(file)
            classes = {**classes, **data}

    dataset = [(os.path.join(dataset_path, file), cls)
               for [file, cls] in list(classes.items())]

    files = [x[0] for x in dataset]
    target = [x[1] for x in dataset]
    features = np.empty(input_size, float)

    for idx, file in enumerate(files):
        if idx % 500 == 0:
            print(f"progress: {idx}/{len(files)}")
        try:
            extracted_features = process_input(file)
            features = np.append(features, [extracted_features], axis=0)
        except Exception as e:
            print(e)

    prepared = zip(files, features, target)
    save_dataset(output_path, prepared)


def save_dataset(dataset_path, dataset):
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)


def load_dataset(dataset_path):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def train(dataset, model_path, ratio, threads, **kwargs):

    dataset = tuple(dataset)

    features = [d[1] for d in dataset]
    classes = [d[2] for d in dataset]

    x_train, x_test, y_train, y_test = train_test_split(
        features, classes, test_size=ratio, random_state=4, stratify=classes)

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x_train)
    train_features_scaled = scaler.transform(x_train)

    classifier = OneVsRestClassifier(SVC(**kwargs), n_jobs=threads)
    classifier.fit(train_features_scaled, y_train)
    save_model(model_path, scaler, classifier)
    print(f"Fitted and saved model to {model_path}")

    test_features_scaled = scaler.transform(x_test)

    print(f"Calculating scores...")

    train_score = classifier.score(train_features_scaled, y_train)
    print(f"Score (train (no. samples {len(y_train)})): {train_score}")

    test_score = classifier.score(test_features_scaled, y_test)
    print(f"Score (test (no. samples {len(y_test)})): {test_score}")


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

    hist = cv2.calcHist([image], [0, 1, 2], None, [
                        16, 16, 16], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None)
    hist = hist.flatten()
    return hist


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='SVM Music Notation classifier')
    parser.add_argument("command", help="subcommand to run", choices=[
                        'train', 'predict', 'build-dataset'])
    args = parser.parse_args(sys.argv[1:2])
    if args.command == "train":
        sub_parser = argparse.ArgumentParser()
        sub_parser.add_argument("--dataset", dest="dataset", required=True)
        sub_parser.add_argument("--output", dest="output", required=True)
        sub_parser.add_argument("--ratio", dest="ratio",
                                type=float, default=0.33)
        sub_parser.add_argument(
            "--coeffc", dest="coeffc", type=float, default=1.0)
        sub_parser.add_argument(
            "--coeff0", dest="coeff0", type=float, default=0.0)
        sub_parser.add_argument("--kernel", dest="kernel", default="rbf")
        sub_parser.add_argument("--gamma", dest="gamma", default="scale")
        sub_parser.add_argument(
            "--tolerance", dest="tolerance", type=float, default=0.001)
        sub_parser.add_argument(
            "--cache_size", dest="cache_size", type=int, default=8000)
        sub_parser.add_argument(
            "--threads", dest="threads", type=int, default=32)
        sub_parser.add_argument(
            "--max_iter", dest="max_iter", type=int, default=-1)
        sub_args = sub_parser.parse_args((sys.argv[2:]))
        return args.command, sub_args
    elif args.command == "predict":
        sub_parser = argparse.ArgumentParser()
        sub_parser.add_argument("--interactive", dest="interactive")
        sub_parser.add_argument("--image", dest="image")
        sub_parser.add_argument("--model", dest="model", required=True)
        sub_args = sub_parser.parse_args((sys.argv[2:]))
        return args.command, sub_args
    elif args.command == "build-dataset":
        sub_parser = argparse.ArgumentParser()
        sub_parser.add_argument(
            "--raw-dataset", dest="raw_dataset", required=True)
        sub_parser.add_argument("--output", dest="output", required=True)
        sub_args = sub_parser.parse_args((sys.argv[2:]))
        return args.command, sub_args
    else:
        print('Unrecognized command')
        parser.print_help()
        exit(1)


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
        dataset = load_dataset(args.dataset)
        train(dataset, args.output, args.ratio, args.threads,
              C=args.coeffc, coef0=args.coeff0, kernel=args.kernel, gamma=args.gamma,
              tol=args.tolerance, cache_size=args.cache_size, max_iter=args.max_iter)
    elif command == "predict":
        predict(args.model, args)
    elif command == "build-dataset":
        prepare_dataset(args.raw_dataset, args.output)


if __name__ == "__main__":
    main()
