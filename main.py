from synthetics.sent_pair import generate_synthetic_sent_pair
from tasks import run_stance_detection, infer_stance_detection, run_synthetic_classification, infer_synthetic_classification


def stance_detection():
    run_stance_detection()
    best_output_path = run_stance_detection()
    infer_stance_detection(
        infer_model_path=best_output_path,
        infer_file="data/stance.test"
    )


def synthetic_classification():
    generate_synthetic_sent_pair()
    best_output_path = run_synthetic_classification()
    # infer_stance_detection(
    #     infer_model_path=best_output_path,
    #     infer_file="data/synthetic.test"
    # )


if __name__ == "__main__":
    stance_detection()
    # synthetic_classification()

