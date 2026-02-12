#!/usr/bin/env python3
"""
GPM Orchestrator
Coordinates multi-agent workflow for poetry generator training
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

from agents.data_agent import DataAgent
from agents.data_preparation_agent import DataPreparationAgent
from agents.validation_agent import ValidationAgent
from agents.training_agent import TrainingAgent

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/gpm_orchestrator.log"),
    ],
)
logger = logging.getLogger("GPM")

PHASE_TO_AGENT = {
    "data": "data_agent",
    "prepare": "data_preparation_agent",
    "validation": "validation_agent",
    "training": "training_agent",
}


class GPMOrchestrator:
    def __init__(self, config_path: str = "config/gpm_config.yaml"):
        self.config = self._load_config(config_path)
        self.results = {}

        for dir_name in ["data", "logs", "checkpoints", "models"]:
            Path(dir_name).mkdir(exist_ok=True)

    def _load_config(self, path: str) -> Dict:
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f)
        except Exception:
            return {
                "ollama": {"model": "llama3.2:3b"},
                "training": {
                    "base_model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
                    "iterations": 2000,
                },
            }

    def run_phase(self, phase_name: str, agent_class, input_data: Dict = None) -> Dict:
        logger.info("=" * 50)
        logger.info(f"PHASE: {phase_name}")
        logger.info("=" * 50)

        try:
            agent = agent_class(self.config)
            result = agent.run(input_data or {})
            self.results[phase_name] = result
            logger.info(f"Phase {phase_name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Phase {phase_name} failed: {e}")
            raise

    def run_full_pipeline(self):
        data_result = self.run_phase("data", DataAgent)
        prepare_result = self.run_phase(
            "prepare", DataPreparationAgent, {"corpus_file": data_result["corpus_file"]}
        )
        validation_result = self.run_phase(
            "validation", ValidationAgent, {"training_file": prepare_result["training_file"]}
        )
        training_result = self.run_phase("training", TrainingAgent, validation_result)
        self._generate_report()
        return training_result

    def run_test_pipeline(self):
        """Quick test: 10 random poems, few synthetic pairs, minimal training."""
        logger.info("Running TEST pipeline (10 random poems, 5 synthetic, 2 style, 50 iters)")
        data_result = self.run_phase("data", DataAgent)
        prepare_input = {
            "corpus_file": data_result["corpus_file"],
            "reverse_limit": 10,
            "synthetic_count": 5,
            "style_topics": ["nature"],
            "style_poet_limit": 2,
            "use_random_poems": True,
        }
        prepare_result = self.run_phase("prepare", DataPreparationAgent, prepare_input)
        validation_result = self.run_phase(
            "validation", ValidationAgent, {"training_file": prepare_result["training_file"]}
        )
        training_input = {
            **validation_result,
            "iterations": 50,
            "adapter_path": "gpm_lora_test",
        }
        training_result = self.run_phase("training", TrainingAgent, training_input)
        self._generate_report()
        logger.info("Test pipeline complete. Adapter: models/adapters/gpm_lora_test")
        return training_result

    def resume_from_checkpoint(self):
        logger.info("Resuming from checkpoint...")
        phases = ["data", "prepare", "validation", "training"]
        last_complete = None

        for phase in phases:
            agent_name = PHASE_TO_AGENT[phase]
            state_file = Path(f"checkpoints/{agent_name}_state.json")
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                    if state.get("status") == "completed":
                        last_complete = phase

        logger.info(f"Last completed phase: {last_complete}")

        if last_complete is None:
            return self.run_full_pipeline()
        if last_complete == "data":
            return self._run_from("prepare")
        if last_complete == "prepare":
            return self._run_from("validation")
        if last_complete == "validation":
            return self._run_from("training")
        return self.run_full_pipeline()

    def _run_from(self, start_phase: str):
        if start_phase == "prepare":
            with open("checkpoints/data_agent_state.json") as f:
                input_data = {"corpus_file": json.load(f)["output_file"]}
        elif start_phase == "validation":
            with open("checkpoints/data_preparation_agent_state.json") as f:
                prep_result = json.load(f)
                input_data = {"training_file": prep_result["output_file"]}
        elif start_phase == "training":
            with open("checkpoints/validation_agent_state.json") as f:
                input_data = {"validated_file": json.load(f)["output_file"]}
        else:
            input_data = {}

        if start_phase == "prepare":
            input_data = self.run_phase("prepare", DataPreparationAgent, input_data)
        if start_phase in ("prepare", "validation"):
            input_data = self.run_phase("validation", ValidationAgent, input_data)
        if start_phase in ("prepare", "validation", "training"):
            input_data = self.run_phase("training", TrainingAgent, input_data)

        self._generate_report()
        return input_data

    def _generate_report(self):
        from datetime import datetime

        report = {
            "project": "Good Poetry Model (GPM) — Poetry Generator",
            "timestamp": datetime.now().isoformat(),
            "phases": self.results,
            "summary": {
                "total_poems": self.results.get("data", {}).get("total_poems", 0),
                "total_pairs": self.results.get("prepare", {}).get("total_pairs", 0),
                "valid_pairs": self.results.get("validation", {}).get("valid_count", 0),
                "final_model": self.results.get("training", {}).get("adapter_path", "N/A"),
            },
        }
        report_file = Path("logs/gpm_final_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {report_file}")
        for key, val in report["summary"].items():
            logger.info(f"  {key}: {val}")


def main():
    parser = argparse.ArgumentParser(description="GPM Orchestrator — Poetry Generator")
    parser.add_argument(
        "--phase",
        choices=["data", "prepare", "validate", "train", "full"],
        default="full",
        help="Run specific phase or full pipeline",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--config", default="config/gpm_config.yaml", help="Config file path"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test: 10 random poems, 5 synthetic pairs, 2 style imitations, 50 training iters",
    )
    args = parser.parse_args()

    orchestrator = GPMOrchestrator(args.config)

    if args.test:
        orchestrator.run_test_pipeline()
    elif args.resume:
        orchestrator.resume_from_checkpoint()
    elif args.phase == "full":
        orchestrator.run_full_pipeline()
    elif args.phase == "data":
        orchestrator.run_phase("data", DataAgent)
    elif args.phase == "prepare":
        with open("checkpoints/data_agent_state.json") as f:
            data_result = json.load(f)
        orchestrator.run_phase(
            "prepare", DataPreparationAgent, {"corpus_file": data_result["output_file"]}
        )
    elif args.phase == "validate":
        with open("checkpoints/data_preparation_agent_state.json") as f:
            prep_result = json.load(f)
        orchestrator.run_phase(
            "validation",
            ValidationAgent,
            {"training_file": prep_result["output_file"]},
        )
    elif args.phase == "train":
        with open("checkpoints/validation_agent_state.json") as f:
            val_result = json.load(f)
        orchestrator.run_phase(
            "training", TrainingAgent, {"validated_file": val_result["output_file"]}
        )


if __name__ == "__main__":
    main()
