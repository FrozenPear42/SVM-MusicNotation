import sys
import pickle
import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels
from joblib import Parallel, delayed
from .svm_preprocessor import process_input, load_dataset, prepare_dataset


def train(dataset, model_path, ratio, **kwargs):
    dataset = tuple(dataset)
    features = [d[1] for d in dataset]
    classes = [d[2] for d in dataset]

    x_train, x_test, y_train, y_test = train_test_split(
        features, classes, test_size=ratio, stratify=classes)

    pipe_estimator = Pipeline([
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('ovr', OneVsRestClassifier(SVC(**kwargs), n_jobs=-1))
    ])

    pipe_estimator.fit(x_train, y_train)
    save_model(model_path, pipe_estimator)
    print(f"Fitted and saved model to {model_path}")

    print(f"Calculating scores...")
    train_score = score_function(pipe_estimator, x_train, y_train)
    print(f"Score (train (no. samples {len(y_train)})): {train_score}")

    test_score = score_function(pipe_estimator, x_test, y_test)
    print(f"Score (test (no. samples {len(y_test)})): {test_score}")

    return pipe_estimator, train_score, test_score


def _predict(estimator, X, start, stop):
    return estimator.predict((X[start:stop]))


def score_function(clf, X, y):

    batches_per_job = 3
    n_jobs = 16
    n_batches = batches_per_job * n_jobs
    n_samples = len(X)
    batch_size = int(np.ceil(n_samples / n_batches))
    parallel = Parallel(n_jobs=n_jobs)
    results = parallel(delayed(_predict)(clf, X, i, i + batch_size)
                       for i in range(0, n_samples, batch_size))
    y_pred = np.concatenate(results)

    c_matrix_acc = confusion_matrix(y, y_pred, normalize='all')
    c_matrix_rel = confusion_matrix(y, y_pred, normalize='true')
    labels = unique_labels(y, y_pred)

    accuracy_all = list(zip(labels, np.diagonal(c_matrix_acc)))
    accuracy_rel = list(zip(labels, np.diagonal(c_matrix_rel)))

    return {
        'correct': np.sum([value for (label, value) in accuracy_all]),
        **{label: value for (label, value) in accuracy_rel},
        # **{labels[coord[0]]+"-" + labels[coord[1]]: x for coord, x in np.ndenumerate(c_matrix)}
    }


def grid_search(dataset, param_grid, cv=5, n_jobs=-1):
    dataset = tuple(dataset)

    features = [d[1] for d in dataset]
    classes = [d[2] for d in dataset]

    pipe_estimator = Pipeline([
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('ovr', OneVsRestClassifier(SVC(), n_jobs=-1))
    ])
    print(param_grid)
    parsed_param_grid = [
        {("ovr__estimator__" + k): v for (k, v) in bundle.items()}
        for bundle in param_grid]

    search_estimator = GridSearchCV(
        pipe_estimator,
        parsed_param_grid,
        n_jobs=n_jobs,
        cv=cv,
        verbose=10,
        scoring=score_function,
        refit='correct',
        return_train_score=False
    )
    search_estimator.fit(features, classes)
    return search_estimator


def predict(model_path, args):
    classifier = load_model(model_path)
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
                prediction = classifier.predict(features)
                print(prediction)
            except Exception as e:
                print(f"error {e}")
    else:
        features = process_input(args.image)
        prediction = classifier.predict(features)
        print(f"predicted class: {prediction}")


def load_model(file_path):
    with open(file_path, "rb") as f:
        classifier = pickle.load(f)
        return classifier


def save_model(file_path, classifier):
    with open(file_path, "wb") as f:
        pickle.dump(classifier, f)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='SVM Music Notation classifier')
    parser.add_argument("command", help="subcommand to run", choices=[
                        'train', 'predict', 'build-dataset', 'grid-search'])
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
    elif args.command == "grid-search":
        sub_parser = argparse.ArgumentParser()
        sub_parser.add_argument(
            "--dataset", dest="dataset", required=True)
        sub_parser.add_argument("--output", dest="output", required=True)
        sub_args = sub_parser.parse_args((sys.argv[2:]))
        return args.command, sub_args
    else:
        print('Unrecognized command')
        parser.print_help()
        exit(1)


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

    elif command == "grid-search":
        dataset = load_dataset(args.dataset)
        param_grid = [{'C': [10000], 'gamma': [1], 'kernel': ['poly']}]
        model_grid = grid_search(dataset, param_grid, cv=3, n_jobs=4)
        save_model(model_grid, args.output)


if __name__ == "__main__":
    main()
