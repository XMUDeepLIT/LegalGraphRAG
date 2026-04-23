import argparse
import os
import subprocess
import sys
from typing import List

from dotenv import load_dotenv


def run_step(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


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
    args = parser.parse_args()

    load_dotenv(dotenv_path=args.dotenv_path)

    final_test = os.path.join(args.raw_data_dir, "final_test.json")
    criminal_law_processed = os.path.join(args.raw_data_dir, "criminal_law_processed.json")
    judicial_explanations = os.path.join(args.raw_data_dir, "judicial_explanations.json")
    base_law_to_crime = os.path.join(args.raw_data_dir, "law_to_crime.json")
    law_corpus = os.path.join(args.raw_data_dir, "law_corpus.jsonl")

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
    run_step(
        [
            py,
            "scripts/prepare_cail_data.py",
            "--input",
            final_test,
            "--max-per-charge",
            str(args.max_per_charge),
            "--cases-output",
            args.cases_base_output,
        ]
    )
    run_step(
        [
            py,
            "scripts/prepare_case_features.py",
            "--dotenv-path",
            args.dotenv_path,
            "--input",
            args.cases_base_output,
            "--output",
            args.cases_output,
        ]
    )
    run_step(
        [
            py,
            "scripts/prepare_law_judge_dep.py",
            "--dotenv-path",
            args.dotenv_path,
            "--criminal-law-processed",
            criminal_law_processed,
            "--judicial-explanations",
            judicial_explanations,
            "--output",
            args.law_judge_dep_output,
        ]
    )
    run_step(
        [
            py,
            "scripts/prepare_law_to_crime.py",
            "--base-law-to-crime",
            base_law_to_crime,
            "--law-judge-dep",
            args.law_judge_dep_output,
            "--law-corpus",
            law_corpus,
            "--output",
            args.law_to_crime_output,
        ]
    )

    print("\nData preparation completed.")
    print(f"- cases: {args.cases_output}")
    print(f"- law_judge_dep: {args.law_judge_dep_output}")
    print(f"- law_to_crime: {args.law_to_crime_output}")


if __name__ == "__main__":
    main()
