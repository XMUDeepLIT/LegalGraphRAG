import argparse
import os
import json
from tqdm import tqdm
import multiprocessing
import time
from typing import List, Dict, Any, Optional

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig, ModelConfig, DataConfig


def load_test_cases(datasets: str, datasets_path: str = "./datasets") -> List[Dict[str, Any]]:
    case_file = os.path.join(datasets_path, f"crime_data_{datasets}_small.json")
    
    if not os.path.exists(case_file):
        raise FileNotFoundError(f"Test dataset not found: {case_file}")
    
    with open(case_file, "r", encoding="utf-8") as f:
        cases = json.load(f)
    
    return cases


def process_cases_worker(
    cases: List[Dict[str, Any]],
    config_dict: Dict[str, Any],
    device: str,
    output_file: str,
    model_name: str
):
    config = LegalGraphRAGConfig.from_dict(config_dict)
    
    config.model.device = device
    config.model.model_name = model_name
    
    rag = LegalGraphRAG(config=config)
    
    results = []
    correct_count = 0
    
    try:
        for case in tqdm(cases, desc=f"Processing on {device} with {model_name}"):
            fact = case.get("fact", "")
            true_charge = case.get("crime", "")
            law_article = case.get("laws", [])
            term_of_imprisonment = case.get("term_of_imprisonment", {})
            
            case_res = rag.analyze_case(case)
            
            results.append({
                "id": case.get("id"),
                "fact": fact,
                "true_charge": true_charge,
                "judge_res": case_res,
                "law_article": law_article,
                "term_of_imprisonment": term_of_imprisonment,
            })
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return correct_count, len(cases)
    
    finally:
        if hasattr(rag, 'model') and hasattr(rag.model, 'release_model'):
            try:
                rag.model.release_model()
            except:
                pass


def run_evaluation(
    model_name: str,
    datasets: str = "CAIL",
    dotenv_path: str = ".env",
    devices: Optional[List[str]] = None,
    datasets_path: str = "./datasets",
    build_graph: bool = True,
    force_rebuild: bool = False
):
    config = LegalGraphRAGConfig.from_env_file(dotenv_path)
    
    output_dir = os.path.join(config.data.output_dir, datasets)
    os.makedirs(output_dir, exist_ok=True)
    
    # Build graph before starting parallel processes
    if build_graph:
        print("="*60)
        print("Building graph database...")
        print("="*60)
        
        # Use first device for graph construction (if devices specified), otherwise use config device
        if devices and len(devices) > 0:
            build_device = devices[0]
        else:
            build_device = config.model.device
        
        # Create configuration for graph construction (using first device)
        build_config = LegalGraphRAGConfig.from_dict(config.to_dict())
        build_config.model.device = build_device
        build_config.model.model_name = model_name
        
        # Create LegalGraphRAG instance and build graph
        print(f"Using device {build_device} for graph construction...")
        rag_builder = LegalGraphRAG(config=build_config)
        rag_builder.build_graph(force_rebuild=force_rebuild)
        
        # Release model resources used for graph construction
        if hasattr(rag_builder, 'model') and hasattr(rag_builder.model, 'release_model'):
            try:
                rag_builder.model.release_model()
            except:
                pass
        
        print("="*60)
        print("Graph database ready!")
        print("="*60)
        print()
    
    test_cases = load_test_cases(datasets, datasets_path)
    print(f"Loaded {len(test_cases)} test cases from {datasets} dataset")
    
    if devices is None:
        devices = ["cuda:2", "cuda:3"]
    if not devices or len(devices) == 0:
        raise ValueError("At least one device must be specified")
    num_processes = len(devices)
    
    chunks = [[] for _ in range(num_processes)]
    for i, case in enumerate(test_cases):
        chunk_index = i % num_processes
        chunks[chunk_index].append(case)
    
    print(f"Split {len(test_cases)} cases into {num_processes} processes")
    for i, chunk in enumerate(chunks):
        print(f"  Process {i} ({devices[i]}): {len(chunk)} cases")
    
    config_dict = config.to_dict()
    
    pool = multiprocessing.Pool(processes=num_processes)
    async_results = []
    
    time_before = time.time()
    
    for i, chunk in enumerate(chunks):
        output_file = f"{model_name}_results_part_{i}.json"
        async_results.append(
            pool.apply_async(
                process_cases_worker,
                args=(
                    chunk,
                    config_dict,
                    devices[i],
                    output_file,
                    model_name,
                ),
            )
        )
    
    pool.close()
    pool.join()
    
    time_after = time.time()
    elapsed_time = time_after - time_before
    
    total_correct = 0
    total_cases = 0
    for res in async_results:
        correct, count = res.get()
        total_correct += correct
        total_cases += count
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {datasets}")
    print(f"Total cases processed: {total_cases}")
    print(f"Correctly classified: {total_correct}/{total_cases}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"{'='*60}\n")
    
    combined_results = []
    for i in range(len(chunks)):
        part_file = f"{model_name}_results_part_{i}.json"
        if os.path.exists(part_file):
            with open(part_file, "r", encoding="utf-8") as f:
                part_data = json.load(f)
                combined_results.extend(part_data)
            os.remove(part_file)
    
    combined_file = os.path.join(output_dir, f"{model_name}_results_combined.json")
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=2)
    
    print(f"Combined results saved to {combined_file}")
    
    stats_file = os.path.join(output_dir, f"{model_name}_stats.json")
    stats = {
        "model_name": model_name,
        "dataset": datasets,
        "total_cases": total_cases,
        "correct_count": total_correct,
        "elapsed_time": elapsed_time,
        "output_file": combined_file
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistics saved to {stats_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Legal Case Analysis with Different Models using LegalGraphRAG"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "qwen3",
            "qwen2_5",
            "gemma3",
            "internlm3",
            "glm4",
            "deepseek_v3",
            "gpt4o_mini",
        ],
        help="Model to use for analysis",
    )
    parser.add_argument(
        "--dotenv_path",
        type=str,
        default=".env",
        help="Path to the .env file",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="CAIL",
        help="Dataset name (e.g., CAIL)",
    )
    parser.add_argument(
        "--datasets_path",
        type=str,
        default=None,
        help="Path to datasets directory (default: ./datasets)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=None,
        help="GPU devices to use (e.g., cuda:2 cuda:3)",
    )
    parser.add_argument(
        "--no-build-graph",
        action="store_true",
        help="Skip graph construction (assume graph already exists)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild graph even if it already exists",
    )
    
    args = parser.parse_args()
    
    multiprocessing.set_start_method("spawn", force=True)
    
    run_evaluation(
        model_name=args.model,
        datasets=args.datasets,
        dotenv_path=args.dotenv_path,
        devices=args.devices,
        datasets_path=args.datasets_path if args.datasets_path else "./datasets",
        build_graph=not args.no_build_graph,
        force_rebuild=args.force_rebuild
    )
