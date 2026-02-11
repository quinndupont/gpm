from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import json
import logging
from datetime import datetime


class GPMAgent(ABC):
    """Base class for all GPM agents following orchestration patterns"""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = self._setup_logger()
        self.state_file = Path(f"checkpoints/{name}_state.json")
        self.output_dir = Path(f"data/{name}")
        Path("checkpoints").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self):
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        Path("logs").mkdir(exist_ok=True)
        handler = logging.FileHandler(f"logs/{self.name}.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        return logger

    def save_state(self, state: Dict[str, Any]):
        """Save progress for resume capability"""
        state['timestamp'] = datetime.now().isoformat()
        state['agent'] = self.name
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        self.logger.info(f"State saved: {state.get('progress', 'unknown')}")

    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load previous state if exists"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return None

    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """Main execution logic - must be implemented by subclasses"""
        pass

    def run(self, input_data: Any = None) -> Any:
        """Wrapper with error handling and state management"""
        try:
            self.logger.info(f"Starting {self.name} execution")
            previous_state = self.load_state()
            if previous_state:
                self.logger.info(f"Resuming from: {previous_state.get('progress', 'start')}")

            result = self.execute(input_data or {})

            self.logger.info(f"Completed {self.name} successfully")
            return result

        except Exception as e:
            self.logger.error(f"Agent {self.name} failed: {str(e)}", exc_info=True)
            self.save_state({'status': 'failed', 'error': str(e)})
            raise
