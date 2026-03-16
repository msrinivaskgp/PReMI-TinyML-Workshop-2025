import os

from train import main_train
from predict_func import run_prediction


def main():

    base_path = "output/"

    print("Starting CTRF Training...")
    main_train()

    print("Running Predictions...")
    run_prediction(base_path, os.path.join(base_path, "results.csv"))

    print("Experiment Completed.")


if __name__ == "__main__":
    main()