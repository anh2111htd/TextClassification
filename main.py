import argparse
from synthetics.sent_pair import generate_synthetic_sent_pair
from tasks import run_stance_detection, infer_stance_detection, \
    run_synthetic_classification, infer_synthetic_classification


def parse_args():
    parser = argparse.ArgumentParser(description='Text Classification')
    parser.add_argument('-t', '--task', type=str, default="synthetic", help='task name')
    parser.add_argument('-m', '--model', type=str, default="cnn", help='model name')
    return parser.parse_args()


def stance_detection(model_name):
    best_output_path = run_stance_detection(model_name)
    infer_stance_detection(
        model_name=model_name,
        infer_model_path=best_output_path,
        infer_file="data/stance.test"
    )


def synthetic_classification(model_name):
    generate_synthetic_sent_pair()
    best_output_path = run_synthetic_classification(model_name)
    infer_synthetic_classification(
        model_name=model_name,
        infer_model_path=best_output_path,
        infer_file="data/synthetic.test"
    )


if __name__ == "__main__":
    args = parse_args()
    if args.task == "synthetic":
        synthetic_classification(args.model)
    elif args.task == "stance":
        stance_detection(args.model)
