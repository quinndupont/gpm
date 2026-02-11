import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

from .base_agent import GPMAgent


class TrainingAgent(GPMAgent):
    """
    Fine-tunes base model using MLX and QLoRA
    Optimized for Mac Mini M4 (24GB RAM)
    """

    def __init__(self, config: Dict):
        super().__init__('training_agent', config)

        train_config = config.get('training', config)
        self.base_model = train_config.get('base_model', 'mlx-community/Llama-3.2-3B-Instruct-4bit')
        self.output_dir = Path('models/adapters')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_layers = train_config.get('num_layers', 16)
        self.batch_size = train_config.get('batch_size', 1)
        self.iterations = train_config.get('iterations', 1000)
        self.learning_rate = train_config.get('learning_rate', 1e-5)
        self.lora_rank = train_config.get('lora_rank', 8)
        self.lora_alpha = train_config.get('lora_alpha', 16)

    def format_for_training(self, validated_file: str) -> Tuple[Path, Path, Path]:
        """Convert validated annotations to MLX training format"""
        data_dir = Path('data/training/gpm_mlx')
        data_dir.mkdir(parents=True, exist_ok=True)

        formatted = []
        with open(validated_file) as f:
            for line in f:
                ann = json.loads(line)

                # Check if this is a ranking task or analysis task
                if 'poem_batch' in ann and isinstance(ann.get('parsed_scores'), list):
                    # Ranking task: use the prompt and raw_response
                    prompt = ann.get('prompt', '')
                    response = ann.get('raw_response', '')
                    if prompt and response:
                        messages = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response}
                        ]
                        formatted.append({"messages": messages})
                elif 'poem' in ann:
                    # Analysis task: use poem and analysis
                    analysis = ann.get('analysis', '')
                    if analysis:
                        messages = [
                            {"role": "user", "content": f"Analyze this poem:\n\n{ann['poem']}"},
                            {"role": "assistant", "content": analysis}
                        ]
                        formatted.append({"messages": messages})

        split_idx = int(len(formatted) * 0.9)
        train_data = formatted[:split_idx]
        val_data = formatted[split_idx:]

        train_file = data_dir / 'train.jsonl'
        val_file = data_dir / 'valid.jsonl'
        with open(train_file, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
        with open(val_file, 'w') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')

        return train_file.parent, train_file, val_file

    def execute(self, input_data: Dict) -> Dict:
        """Run QLoRA fine-tuning with MLX"""
        validated_file = input_data.get('validated_file', 'data/training/gpm_validated.jsonl')

        data_dir, train_file, val_file = self.format_for_training(validated_file)
        self.logger.info(f"Training data: {train_file}")
        self.logger.info(f"Validation data: {val_file}")

        adapter_path = self.output_dir / 'gpm_lora'
        adapter_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("Starting fine-tuning with mlx_lm.lora CLI...")
        cmd = [
            'python', '-m', 'mlx_lm.lora',
            '--model', self.base_model,
            '--train',
            '--data', str(data_dir),
            '--iters', str(self.iterations),
            '--batch-size', str(self.batch_size),
            '--learning-rate', str(self.learning_rate),
            '--adapter-path', str(adapter_path),
            '--num-layers', str(self.num_layers),
            '--steps-per-report', '50',
            '--save-every', '100',
            '--max-seq-length', '2048',
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        err_lines = []
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            err_lines.append(line)
        returncode = proc.wait()
        if returncode != 0:
            stderr = "".join(err_lines)
            self.logger.error(f"Training failed: {stderr}")
            raise RuntimeError(f"mlx_lm.lora failed: {stderr}")

        self.save_state({
            'status': 'completed',
            'base_model': self.base_model,
            'adapter_path': str(adapter_path),
        })

        return {
            'adapter_path': str(adapter_path),
            'base_model': self.base_model,
            'status': 'trained'
        }
