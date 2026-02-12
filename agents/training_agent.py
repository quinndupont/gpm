import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from .base_agent import GPMAgent


SYSTEM_PROMPT = """You are a gifted poet who writes original, evocative poetry. You write in various forms and styles, from formal sonnets to experimental free verse. Your poems feature precise imagery, emotional authenticity, and attention to sound and rhythm."""


def _get_tokenizer(base_model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)


def _input_ids_len(out) -> int:
    """Length of token ids from apply_chat_template (handles BatchEncoding or list)."""
    if hasattr(out, "input_ids"):
        ids = out["input_ids"]
    else:
        ids = out
    if isinstance(ids, list) and ids and isinstance(ids[0], int):
        return len(ids)
    return sum(len(s) for s in ids) if ids else 0


def _token_count(tokenizer, messages: List[Dict]) -> int:
    try:
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        return _input_ids_len(out)
    except Exception:
        text = " ".join(m.get("content", "") for m in messages)
        return len(tokenizer.encode(text, add_special_tokens=True))


def _split_long_messages(
    tokenizer, messages: List[Dict], max_tokens: int
) -> List[Dict]:
    """Split messages into examples if over max_tokens. Supports [user, assistant] or [system, user, assistant]."""
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return [{"messages": messages}]
    assistant_msg = messages[-1]
    prefix_messages = messages[:-1]
    content = assistant_msg.get("content", "") or ""
    if not content.strip():
        return [{"messages": messages}]

    full_count = _token_count(tokenizer, messages)
    if full_count <= max_tokens:
        return [{"messages": messages}]

    try:
        prefix_out = tokenizer.apply_chat_template(
            prefix_messages, tokenize=True, add_generation_prompt=True
        )
        prefix_len = _input_ids_len(prefix_out)
    except Exception:
        prefix_len = sum(
            len(tokenizer.encode(m.get("content", ""), add_special_tokens=False))
            for m in prefix_messages
        )

    reserve = 100
    if prefix_len >= max_tokens - reserve:
        asst_ids = tokenizer.encode(content, add_special_tokens=False)
        room = max(50, max_tokens - prefix_len - reserve)
        truncated_ids = asst_ids[:room] if room > 0 else asst_ids[:100]
        truncated_content = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        truncated_asst = {"role": "assistant", "content": truncated_content.strip() or " "}
        return [{"messages": prefix_messages + [truncated_asst]}]

    max_assistant_tokens = max(100, max_tokens - prefix_len - 8)
    asst_ids = tokenizer.encode(content, add_special_tokens=False)
    if len(asst_ids) <= max_assistant_tokens:
        return [{"messages": messages}]

    out = []
    for start in range(0, len(asst_ids), max_assistant_tokens):
        chunk_ids = asst_ids[start : start + max_assistant_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunk_messages = prefix_messages + [
            {"role": "assistant", "content": chunk_text.strip() or " "},
        ]
        while _token_count(tokenizer, chunk_messages) > max_tokens and len(chunk_ids) > 50:
            chunk_ids = chunk_ids[: len(chunk_ids) // 2]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunk_messages = prefix_messages + [
                {"role": "assistant", "content": chunk_text.strip() or " "},
            ]
        out.append({"messages": chunk_messages})
    return out


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

        self.num_layers = train_config.get("num_layers", 20)
        self.batch_size = train_config.get("batch_size", 1)
        self.iterations = train_config.get("iterations", 2000)
        self.learning_rate = train_config.get("learning_rate", 2e-5)
        self.lora_rank = train_config.get("lora_rank", 16)
        self.lora_alpha = train_config.get("lora_alpha", 32)
        self.max_seq_length = train_config.get("max_seq_length", 1024)

    def format_for_training(self, validated_file: str) -> Tuple[Path, Path, Path]:
        """Convert validated (prompt, poem) pairs to ChatML format for MLX. Long sequences are split."""
        data_dir = Path("data/training/gpm_mlx")
        data_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Loading tokenizer for length check and pre-splitting...")
        tokenizer = _get_tokenizer(self.base_model)

        formatted = []
        n_split = 0
        with open(validated_file) as f:
            for line in f:
                pair = json.loads(line)
                prompt = pair.get("prompt", "")
                poem = pair.get("poem", "")
                if not prompt or not poem:
                    continue

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": poem},
                ]
                items = _split_long_messages(tokenizer, messages, self.max_seq_length)
                if len(items) > 1:
                    n_split += 1
                formatted.extend(items)

        if n_split:
            self.logger.info(f"Pre-split {n_split} long example(s) to stay under {self.max_seq_length} tokens.")

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

        iterations = input_data.get("iterations", self.iterations)
        adapter_name = input_data.get("adapter_path", "gpm_lora")
        adapter_path = self.output_dir / adapter_name
        adapter_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("Starting fine-tuning with mlx_lm.lora CLI...")
        cmd = [
            "python",
            "-m",
            "mlx_lm.lora",
            "--model",
            self.base_model,
            "--train",
            "--data",
            str(data_dir),
            "--iters",
            str(iterations),
            "--batch-size",
            str(self.batch_size),
            "--learning-rate",
            str(self.learning_rate),
            "--adapter-path",
            str(adapter_path),
            "--num-layers",
            str(self.num_layers),
            "--steps-per-report",
            "50",
            "--save-every",
            "100",
            "--max-seq-length",
            str(self.max_seq_length),
            "--mask-prompt",
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
