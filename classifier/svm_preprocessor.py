import glob
import json
import cv2
import os
import mahotas
import pickle
import numpy as np

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
