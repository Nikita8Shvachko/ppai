"""
AI Training Manager

Integrated training system using ParallelAIController with deep training
capabilities and human-like AI difficulty support.
"""

import os
import time
from typing import Dict
from enum import Enum

from ai_difficulty_presets import AIDifficultyPresets
from parallel_ai_controller import ParallelAIController


class TrainingMode(Enum):
    """
    Available training modes with different intensities.
    """

    STANDARD = {"name": "Standard Training", "episodes": 1000, "agents": 4}
    DEEP = {"name": "Deep Training", "episodes": 5000, "agents": 8}
    INTENSIVE = {"name": "Intensive Training", "episodes": 10000, "agents": 16}


class TrainingStatus(Enum):
    """
    Training process status states.
    """

    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    ERROR = "error"


class AITrainingManager:
    """
    Manages AI training using the integrated ParallelAIController system.
    Supports deep training with different difficulty presets.
    """

    def __init__(self):
        """
        Initialize the AITrainingManager.
        Returns:
            None
        """
        self.models_dir = "ai_models"
        os.makedirs(self.models_dir, exist_ok=True)

        self.training_status = TrainingStatus.IDLE
        self.training_progress = 0.0
        self.training_thread = None
        self.current_log_messages = []
        self.max_log_messages = 50

        self.current_ai_controller = None
        self.current_difficulty_preset = None
        self.current_training_mode = None
        self.training_start_time = None
        self.episodes_completed = 0
        self.target_episodes = 0

    def get_available_difficulty_presets(self):
        """
        Get all available AI difficulty presets for training.
        Returns:
            List[Dict]: The available difficulty presets
        """
        return AIDifficultyPresets.get_all_presets()

    def get_training_modes(self):
        """
        Get all available training modes.
        Returns:
            List[Dict]: The available training modes
        """
        return [mode.value for mode in TrainingMode]

    def start_training(
        self,
        difficulty_preset: Dict,
        training_mode: Dict,
        paddle=None,
        ball=None,
        sync: bool = False,
    ) -> bool:
        """
        Start AI training with specified difficulty and training mode.

        Args:
            difficulty_preset: AI difficulty settings from AIDifficultyPresets
            training_mode: Training configuration (episodes, agents, etc.)
            paddle: Paddle object for training
            ball: Ball object for training
            sync: Whether to run training synchronously

        Returns:
            bool: True if training started successfully
        """
        try:
            if self.training_status != TrainingStatus.IDLE:
                self._log("Training already in progress")
                return False

            if not paddle or not ball:
                self._log("Error: Paddle and ball objects required for training")
                return False

            self.current_difficulty_preset = difficulty_preset
            self.current_training_mode = training_mode
            self.target_episodes = training_mode["episodes"]
            self.episodes_completed = 0

            return self._train_sync(paddle, ball)

        except Exception as e:
            self._log(f"Error starting training: {e}")
            self.training_status = TrainingStatus.ERROR
            return False

    def _train_sync(self, paddle, ball) -> bool:
        """
        Execute training synchronously.
        Returns:
            bool: True if training started successfully
        """
        try:
            self.training_status = TrainingStatus.PREPARING
            self.training_progress = 0.0
            self.training_start_time = time.time()

            difficulty_name = self.current_difficulty_preset.get("name", "Unknown")
            mode_name = self.current_training_mode.get("name", "Unknown")

            self._log(f"Starting {mode_name}")
            self._log(f"AI Difficulty: {difficulty_name}")
            self._log(f"Target Episodes: {self.target_episodes}")

            num_agents = self.current_training_mode["agents"]
            self.current_ai_controller = ParallelAIController(
                paddle=paddle,
                ball=ball,
                num_agents=num_agents,
                input_size=12,
                hidden_size=32,
                output_size=1,
                exchange_interval=200,
                mutation_rate=0.08,
                batch_size=32,
                difficulty_preset=self.current_difficulty_preset,
            )

            self._log(f"Created AI with {num_agents} agents")
            self._log(f"Difficulty applied: {difficulty_name}")

            self.training_status = TrainingStatus.TRAINING
            self.current_ai_controller.start_training()

            episodes_per_update = max(1, self.target_episodes // 100)

            for episode in range(self.target_episodes):
                time.sleep(0.01)

                if episode % episodes_per_update == 0:
                    self.episodes_completed = episode
                    self.training_progress = (episode / self.target_episodes) * 100
                    elapsed_time = time.time() - self.training_start_time

                    if hasattr(self.current_ai_controller, "get_performance_metrics"):
                        metrics = self.current_ai_controller.get_performance_metrics()
                        accuracy = metrics.get("long_term_accuracy", 0.0)
                        self._log(
                            f"Episode {episode}/{self.target_episodes} | "
                            f"Accuracy: {accuracy:.1%} | "
                            f"Time: {elapsed_time:.1f}s"
                        )
                    else:
                        self._log(
                            f"Episode {episode}/{self.target_episodes} | "
                            f"Time: {elapsed_time:.1f}s"
                        )

            self.episodes_completed = self.target_episodes
            self.training_progress = 100.0

            model_filename = self._generate_model_filename()
            model_path = os.path.join(self.models_dir, model_filename)
            self.current_ai_controller.save_model(model_path)

            elapsed_time = time.time() - self.training_start_time
            self._log("Training completed")
            self._log(f"Model saved: {model_filename}")
            self._log(f"Total time: {elapsed_time:.1f}s")

            self.training_status = TrainingStatus.COMPLETED
            return True

        except Exception as e:
            self._log(f"Training error: {e}")
            self.training_status = TrainingStatus.ERROR
            return False
        finally:
            if self.current_ai_controller:
                self.current_ai_controller.stop_training()

    def _train_async(self, paddle, ball):
        """
        Asynchronous training wrapper.
        Returns:
            None
        """
        self._train_sync(paddle, ball)

    def _generate_model_filename(self) -> str:
        """
        Generate filename for trained model.
        Returns:
            str: The filename for the trained model
        """
        difficulty_name = self.current_difficulty_preset.get("name", "unknown")
        mode_name = self.current_training_mode.get("name", "standard")
        timestamp = int(time.time())

        difficulty_clean = (
            difficulty_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        )
        mode_clean = mode_name.lower().replace(" ", "_")

        return f"{difficulty_clean}_{mode_clean}_{timestamp}.pkl"

    def get_training_status(self) -> Dict:
        """
        Get current training status and metrics.
        Returns:
            Dict: The current training status and metrics
        """
        return {
            "status": self.training_status.value,
            "progress": self.training_progress,
            "episodes_completed": self.episodes_completed,
            "target_episodes": self.target_episodes,
            "difficulty": (
                self.current_difficulty_preset.get("name", "None")
                if self.current_difficulty_preset
                else "None"
            ),
            "training_mode": (
                self.current_training_mode.get("name", "None")
                if self.current_training_mode
                else "None"
            ),
            "logs": self.current_log_messages.copy(),
        }

    def cancel_training(self):
        """
        Cancel current training session.
        Returns:
            None
        """
        try:
            if self.current_ai_controller:
                self.current_ai_controller.stop_training()

            self.training_status = TrainingStatus.IDLE
            self.training_progress = 0.0
            self.episodes_completed = 0
            self._log("Training cancelled")
        except Exception as e:
            self._log(f"Error cancelling training: {e}")

    def get_available_models(self):
        """
        Get list of available trained models.
        Returns:
            List[Dict]: The available trained models
        """
        models = []
        try:
            if os.path.exists(self.models_dir):
                for filename in os.listdir(self.models_dir):
                    if filename.endswith(".pkl"):
                        filepath = os.path.join(self.models_dir, filename)
                        if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
                            model_info = self._analyze_model_file(filename, filepath)
                            models.append(model_info)

            models.sort(key=lambda x: x.get("created", 0), reverse=True)

        except Exception as e:
            self._log(f"Error scanning models: {e}")

        return models

    def _analyze_model_file(self, filename: str, filepath: str) -> Dict:
        """
        Analyze model file to extract metadata.
        Returns:
            Dict: The metadata of the model file
        """
        try:
            file_size = os.path.getsize(filepath)
            file_size_kb = file_size // 1024
            created_time = os.path.getctime(filepath)

            if "beginner" in filename.lower():
                difficulty = "Beginner"
                quality_score = 30
            elif "casual" in filename.lower():
                difficulty = "Casual Player"
                quality_score = 60
            elif "skilled" in filename.lower():
                difficulty = "Skilled Player"
                quality_score = 80
            elif "expert" in filename.lower():
                difficulty = "Expert Player"
                quality_score = 95
            elif "unbeatable" in filename.lower():
                difficulty = "Unbeatable"
                quality_score = 100
            else:
                difficulty = "Unknown"
                quality_score = 50

            if "deep" in filename.lower():
                training_type = "Deep Training"
            elif "intensive" in filename.lower():
                training_type = "Intensive Training"
            else:
                training_type = "Standard Training"

            return {
                "filename": filename,
                "filepath": filepath,
                "size_kb": file_size_kb,
                "created": created_time,
                "difficulty": difficulty,
                "training_type": training_type,
                "quality_score": quality_score,
                "display_name": f"{difficulty} - {training_type}",
                "description": f"Trained AI model ({difficulty}, {file_size_kb}KB)",
            }

        except Exception as e:
            self._log(f"Error analyzing model {filename}: {e}")
            return {
                "filename": filename,
                "filepath": filepath,
                "size_kb": 0,
                "created": 0,
                "difficulty": "Unknown",
                "training_type": "Unknown",
                "quality_score": 0,
                "display_name": filename.replace(".pkl", ""),
                "description": "Model analysis failed",
            }

    def delete_model(self, filename: str) -> bool:
        """
        Delete a trained model.
        Returns:
            bool: True if the model was deleted successfully
        """
        try:
            filepath = os.path.join(self.models_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                self._log(f"Deleted model: {filename}")
                return True
            else:
                self._log(f"Model not found: {filename}")
                return False
        except Exception as e:
            self._log(f"Error deleting model {filename}: {e}")
            return False

    def _log(self, message: str):
        """
        Log a message with timestamp.
        Returns:
            None
        """
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)

        self.current_log_messages.append(log_message)
        if len(self.current_log_messages) > self.max_log_messages:
            self.current_log_messages.pop(0)


training_manager = AITrainingManager()
