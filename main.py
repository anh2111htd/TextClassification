from tasks import run_stance_detection, infer_stance_detection


if __name__ == "__main__":
    best_output_path = run_stance_detection()
    infer_stance_detection(
        infer_model_path=best_output_path,
        infer_file="data/stance_better.test"
    )
