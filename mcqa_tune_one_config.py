import argparse
import subprocess
import sys
from pathlib import Path


def run(command):
    print("\n" + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at")
    parser.add_argument("--prefix", default="results_joint_randomized_tune")
    args = parser.parse_args()

    lambda_values = [0.0, 2.0, 4.0]
    num_seeds = 5
    total_tasks = len(lambda_values) ** 3 * num_seeds

    if args.task_id < 0 or args.task_id >= total_tasks:
        raise ValueError(f"task-id must be between 0 and {total_tasks - 1}")

    config_id = args.task_id // num_seeds
    seed = args.task_id % num_seeds

    ap_index = config_id // 16
    remaining = config_id % 16
    c_ap_index = remaining // 4
    c_at_index = remaining % 4

    ap_lambda = lambda_values[ap_index]
    c_ap_lambda = lambda_values[c_ap_index]
    c_at_lambda = lambda_values[c_at_index]

    result_dir = f"{args.prefix}_ap{ap_lambda}_cap{c_ap_lambda}_cat{c_at_lambda}_seed{seed}"
    eval_path = Path(result_dir) / "eval_results.txt"

    print("=" * 100)
    print(f"task-id={args.task_id}")
    print(f"AP={ap_lambda} | C-AP={c_ap_lambda} | C-AT={c_at_lambda} | seed={seed}")
    print(f"result-dir={result_dir}")
    print("=" * 100, flush=True)

    if eval_path.exists():
        print("Already finished. Skip.", flush=True)
        return

    train_command = [sys.executable, "mcqa_joint_das_ap_at_c_balanced.py"]
    train_command += ["--data-path", args.data_path]
    train_command += ["--result-dir", result_dir]
    train_command += ["--seed", str(seed)]
    train_command += ["--ap-lambda-med", str(ap_lambda)]
    train_command += ["--ap-lambda-readout", str(ap_lambda)]
    train_command += ["--c-lambda-ap-med", str(c_ap_lambda)]
    train_command += ["--c-lambda-ap-readout", str(c_ap_lambda)]
    train_command += ["--c-lambda-at-med", str(c_at_lambda)]
    train_command += ["--c-lambda-at-readout", str(c_at_lambda)]

    run(train_command)

    eval_command = [sys.executable, "mcqa_eval_readout_mediator_new.py"]
    eval_command += ["--data-path", args.data_path]
    eval_command += ["--result-dir", result_dir]
    eval_command += ["--seed", str(seed)]

    run(eval_command)


if __name__ == "__main__":
    main()
