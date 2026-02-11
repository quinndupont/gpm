#!/usr/bin/env python3
"""
GPM Orchestrator
Coordinates multi-agent workflow for poetry model training
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from agents.data_agent import DataAgent
from agents.annotation_agent import AnnotationAgent
from agents.validation_agent import ValidationAgent
from agents.training_agent import TrainingAgent

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/gpm_orchestrator.log')
    ]
)
logger = logging.getLogger('GPM')

PHASE_TO_AGENT = {
    'data_preparation': 'data_agent',
    'annotation': 'annotation_agent',
    'validation': 'validation_agent',
    'training': 'training_agent',
}


class GPMOrchestrator:
    def __init__(self, config_path: str = 'config/gpm_config.yaml', test_data_dir: str = None):
        self.config = self._load_config(config_path)
        self.results = {}
        self.test_data_dir = Path(test_data_dir) if test_data_dir else None

        for dir_name in ['data', 'logs', 'checkpoints', 'models']:
            Path(dir_name).mkdir(exist_ok=True)

    def _load_config(self, path: str) -> Dict:
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f)
        except Exception:
            return {
                'ollama': {'model': 'llama3.2:3b', 'batch_size': 5},
                'training': {'base_model': 'mlx-community/Llama-3.2-3B-Instruct-4bit', 'iterations': 1000}
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
        data_result = self.run_phase('data_preparation', DataAgent)

        # Use test annotations if provided, otherwise run annotation
        if self.test_data_dir:
            logger.info(f"Using test annotations from {self.test_data_dir}")
            annotation_result = self._load_test_annotations()
        else:
            annotation_result = self.run_phase('annotation', AnnotationAgent, data_result)

        validation_result = self.run_phase('validation', ValidationAgent, annotation_result)
        training_result = self.run_phase('training', TrainingAgent, validation_result)
        self._generate_report()
        return training_result

    def resume_from_checkpoint(self):
        logger.info("Resuming from checkpoint...")
        phases = ['data_preparation', 'annotation', 'validation', 'training']
        last_complete = None

        for phase in phases:
            agent_name = PHASE_TO_AGENT[phase]
            state_file = Path(f"checkpoints/{agent_name}_state.json")
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                    if state.get('status') == 'completed':
                        last_complete = phase

        logger.info(f"Last completed phase: {last_complete}")

        if last_complete is None:
            return self.run_full_pipeline()
        if last_complete == 'data_preparation':
            return self._run_from('annotation')
        if last_complete == 'annotation':
            return self._run_from('validation')
        if last_complete == 'validation':
            return self._run_from('training')
        return self.run_full_pipeline()

    def _run_from(self, start_phase: str):
        if start_phase == 'annotation':
            with open('checkpoints/data_agent_state.json') as f:
                input_data = {'corpus_file': json.load(f)['output_file']}
        elif start_phase == 'validation':
            # Check if using test data or annotation checkpoint
            if self.test_data_dir:
                input_data = self._load_test_annotations()
            else:
                with open('checkpoints/annotation_agent_state.json') as f:
                    input_data = {'annotations_file': json.load(f)['output_file']}
        elif start_phase == 'training':
            with open('checkpoints/validation_agent_state.json') as f:
                input_data = {'validated_file': json.load(f)['output_file']}
        else:
            input_data = {}

        if start_phase == 'annotation':
            # Use test data if provided, otherwise run annotation
            if self.test_data_dir:
                input_data = self._load_test_annotations()
            else:
                input_data = self.run_phase('annotation', AnnotationAgent, input_data)
        if start_phase in ('annotation', 'validation'):
            input_data = self.run_phase('validation', ValidationAgent, input_data)
        if start_phase in ('annotation', 'validation', 'training'):
            input_data = self.run_phase('training', TrainingAgent, input_data)

        self._generate_report()
        return input_data

    def _load_test_annotations(self) -> Dict:
        """Load pre-existing test annotations from test data directory"""
        annotations_file = self.test_data_dir / 'annotations.jsonl'
        batches_file = self.test_data_dir / 'poem_batches.json'

        if not annotations_file.exists():
            raise FileNotFoundError(f"Test annotations file not found: {annotations_file}")
        if not batches_file.exists():
            raise FileNotFoundError(f"Test batches file not found: {batches_file}")

        # Count annotations
        total_annotated = 0
        with open(annotations_file) as f:
            for line in f:
                if line.strip():
                    total_annotated += 1

        logger.info(f"Loaded {total_annotated} test annotations from {annotations_file}")

        # Store results for report generation (with output_file for backward compatibility)
        self.results['annotation'] = {
            'output_file': str(annotations_file),
            'batches_file': str(batches_file),
            'total_annotated': total_annotated,
            'mode': 'test'
        }

        # Return format expected by ValidationAgent
        return {
            'annotations_file': str(annotations_file),
            'batches_file': str(batches_file),
            'total_annotated': total_annotated
        }

    def _generate_report(self):
        from datetime import datetime
        report = {
            'project': 'Good Poetry Model (GPM)',
            'timestamp': datetime.now().isoformat(),
            'phases': self.results,
            'summary': {
                'total_poems': self.results.get('data_preparation', {}).get('total_poems', 0),
                'total_annotated': self.results.get('annotation', {}).get('total_annotated', 0),
                'valid_annotations': self.results.get('validation', {}).get('valid_count', 0),
                'final_model': self.results.get('training', {}).get('adapter_path', 'N/A')
            }
        }
        report_file = Path('logs/gpm_final_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {report_file}")
        for key, val in report['summary'].items():
            logger.info(f"  {key}: {val}")


def main():
    parser = argparse.ArgumentParser(description='GPM Orchestrator')
    parser.add_argument('--phase', choices=['data', 'annotate', 'validate', 'train', 'full'],
                        default='full', help='Run specific phase or full pipeline')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--config', default='config/gpm_config.yaml', help='Config file path')
    parser.add_argument('--test', type=str, help='Use test annotations from specified directory (e.g., data/annotation_test/20260211_151511)')
    args = parser.parse_args()

    orchestrator = GPMOrchestrator(args.config, test_data_dir=args.test)

    if args.resume:
        orchestrator.resume_from_checkpoint()
    elif args.phase == 'full':
        orchestrator.run_full_pipeline()
    elif args.phase == 'data':
        orchestrator.run_phase('data_preparation', DataAgent)
    elif args.phase == 'annotate':
        if args.test:
            logger.warning("Cannot use --test with --phase annotate (test data skips annotation)")
        else:
            with open('checkpoints/data_agent_state.json') as f:
                data_result = json.load(f)
            orchestrator.run_phase('annotation', AnnotationAgent, {'corpus_file': data_result['output_file']})
    elif args.phase == 'validate':
        if args.test:
            input_data = orchestrator._load_test_annotations()
        else:
            with open('checkpoints/annotation_agent_state.json') as f:
                ann_result = json.load(f)
                input_data = {'annotations_file': ann_result['output_file']}
        orchestrator.run_phase('validation', ValidationAgent, input_data)
    elif args.phase == 'train':
        with open('checkpoints/validation_agent_state.json') as f:
            val_result = json.load(f)
        orchestrator.run_phase('training', TrainingAgent,
                              {'validated_file': val_result['output_file']})


if __name__ == '__main__':
    main()
