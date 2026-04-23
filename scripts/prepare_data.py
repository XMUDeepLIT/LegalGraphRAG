import argparse
import os
import subprocess
import sys
from typing import List

from dotenv import load_dotenv


def to_abs(path: str, project_root: str) -> str:
    return path if os.path.isabs(path) else os.path.join(project_root, path)


def run_step(cmd: List[str], cwd: str) -> None:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print("\nStep failed:", " ".join(cmd))
        raise SystemExit(result.returncode)


def outputs_exist(paths: List[str]) -> bool:
    return all(os.path.exists(p) for p in paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-click data preparation for LegalGraphRAG."
    )
    parser.add_argument(
        "--dotenv-path",
        type=str,
        default=".env",
        help="Path to .env file.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="./raw_data",
        help="Directory containing raw input files.",
    )
    parser.add_argument(
        "--max-per-charge",
        type=int,
        default=5,
        help="Max samples per charge when preparing CAIL cases.",
    )
    parser.add_argument(
        "--cases-output",
        type=str,
        default="./datas/cases_with_feature.json",
        help="Output path for prepared cases with features.",
    )
    parser.add_argument(
        "--cases-base-output",
        type=str,
        default="./datas/cases_base.json",
        help="Intermediate output path for cases before feature extraction.",
    )
    parser.add_argument(
        "--dataset-output",
        type=str,
        default="./datasets/crime_data_CAIL_small.json",
        help="Output path for generated evaluation dataset file.",
    )
    parser.add_argument(
        "--law-judge-dep-output",
        type=str,
        default="./datas/law_judge_dep.json",
        help="Output path for generated law_judge_dep.",
    )
    parser.add_argument(
        "--law-to-crime-output",
        type=str,
        default="./datas/law_to_crime.json",
        help="Output path for enriched law_to_crime.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun all steps even if output files already exist.",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = to_abs(args.dotenv_path, project_root)
    raw_data_dir = to_abs(args.raw_data_dir, project_root)
    cases_output = to_abs(args.cases_output, project_root)
    cases_base_output = to_abs(args.cases_base_output, project_root)
    dataset_output = to_abs(args.dataset_output, project_root)
    law_judge_dep_output = to_abs(args.law_judge_dep_output, project_root)
    law_to_crime_output = to_abs(args.law_to_crime_output, project_root)

    load_dotenv(dotenv_path=dotenv_path)

    final_test = os.path.join(raw_data_dir, "final_test.json")
    criminal_law_processed = os.path.join(raw_data_dir, "criminal_law_processed.json")
    judicial_explanations = os.path.join(raw_data_dir, "judicial_explanations.json")
    base_law_to_crime = os.path.join(raw_data_dir, "law_to_crime.json")
    law_corpus = os.path.join(raw_data_dir, "law_corpus.jsonl")

    required_files = [
        final_test,
        criminal_law_processed,
        judicial_explanations,
        base_law_to_crime,
        law_corpus,
    ]
    missing = [x for x in required_files if not os.path.exists(x)]
    if missing:
        raise FileNotFoundError("Missing raw data files:\n" + "\n".join(missing))

    py = sys.executable
    step_a_outputs = [cases_base_output, dataset_output]
    step_a_ran = False
    if args.force or not outputs_exist(step_a_outputs):
        run_step(
            [
                py,
                "scripts/prepare_cail_data.py",
                "--input",
                final_test,
                "--max-per-charge",
                str(args.max_per_charge),
                "--cases-output",
                cases_base_output,
                "--dataset-output",
                dataset_output,
            ],
            cwd=project_root,
        )
        step_a_ran = True
    else:
        print("Skipping Step A: cases base and dataset already exist.")

    step_a2_ran = False
    if args.force or step_a_ran or not os.path.exists(cases_output):
        run_step(
            [
                py,
                "scripts/prepare_case_features.py",
                "--dotenv-path",
                dotenv_path,
                "--input",
                cases_base_output,
                "--output",
                cases_output,
            ],
            cwd=project_root,
        )
        step_a2_ran = True
    else:
        print("Skipping Step A2: cases_with_feature already exists.")

    step_b_ran = False
    if args.force or not os.path.exists(law_judge_dep_output):
        run_step(
            [
                py,
                "scripts/prepare_law_judge_dep.py",
                "--dotenv-path",
                dotenv_path,
                "--criminal-law-processed",
                criminal_law_processed,
                "--judicial-explanations",
                judicial_explanations,
                "--output",
                law_judge_dep_output,
            ],
            cwd=project_root,
        )
        step_b_ran = True
    else:
        print("Skipping Step B: law_judge_dep already exists.")

    if args.force or step_b_ran or not os.path.exists(law_to_crime_output):
        run_step(
            [
                py,
                "scripts/prepare_law_to_crime.py",
                "--base-law-to-crime",
                base_law_to_crime,
                "--law-judge-dep",
                law_judge_dep_output,
                "--law-corpus",
                law_corpus,
                "--output",
                law_to_crime_output,
            ],
            cwd=project_root,
        )
    else:
        print("Skipping Step C: law_to_crime already exists.")

    print("\nData preparation completed.")
    print(f"- cases: {cases_output}")
    print(f"- dataset: {dataset_output}")
    print(f"- law_judge_dep: {law_judge_dep_output}")
    print(f"- law_to_crime: {law_to_crime_output}")


if __name__ == "__main__":
    main()
