import os
import pickle
import random
import time

import numpy as np

from neural_network import NeuralNetwork


class AIController:
    def __init__(
        self,
        paddle,
        ball,
        input_size=12,
        hidden_size=32,
        output_size=1,
        learning_rate=0.001,
        prediction_time_limit=0.5,
        distance_error_factor=40,
        base_prediction_error=10,
        wall_bounce_accuracy=0.9,
    ):
        """
        Initialize AI controller.

        Args:
            paddle: Paddle object
            ball: Ball object
            input_size: Neural network input layer size
            hidden_size: Hidden layer size
            output_size: Output layer size
            learning_rate: Learning rate
        """
        self.paddle = paddle
        self.ball = ball
        self.max_ball_speed = max(abs(ball.speed_x), abs(ball.speed_y))
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.training = True
        self.agent_id = 0
        self.long_term_episodes = 0

        self.neural_network = NeuralNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            learning_rate=self.learning_rate,
        )

        self.base_learning_rate = 0.01
        self.episodes = 0
        self.successful_hits = 0
        self.total_hits = 0
        self.total_rewards = 0
        self.last_paddle_y = None
        self.consecutive_misses = 0
        self.consecutive_hits = 0

        self.round_successful_hits = 0
        self.round_total_hits = 0
        self.round_consecutive_hits = 0
        self.round_consecutive_misses = 0
        self.round_rewards = 0

        self.long_term_successful_hits = 0
        self.long_term_total_hits = 0
        self.long_term_consecutive_hits = 0
        self.long_term_best_consecutive_hits = 0
        self.long_term_rewards = 0

        self.performance_history = []
        self.max_history_size = 10

        self.current_difficulty = 1.0
        self.difficulty_history = []

        self.last_ball_x = ball.rect.x
        self.last_ball_speed_x = ball.speed_x
        self.ball_in_paddle_zone = False
        self.opportunity_processed = False

        self.is_left_paddle = paddle.rect.x < 200

        self.accuracy_thresholds = [0.3, 0.5, 0.7, 0.9]
        self.saved_thresholds = set()

        self.save_interval = 100

        self.reaction_delay = 0.05
        self.last_reaction_time = 0
        self.mistake_probability = 0.05
        self.prediction_error = 0.05

        self.prediction_time_limit = prediction_time_limit
        self.distance_error_factor = distance_error_factor
        self.base_prediction_error = base_prediction_error
        self.wall_bounce_accuracy = wall_bounce_accuracy

    def get_state(self):
        """
        Get extended game state.
        Universal state for left and right paddle.

        Returns:
            numpy.ndarray: Normalized game state
        """
        if not self.ball or not hasattr(self.ball, "rect"):
            return np.array([0.5] * self.input_size).reshape(1, -1)

        try:
            ball_x = self.ball.rect.x / self.paddle.game.WIDTH
            ball_y = self.ball.rect.y / self.paddle.game.HEIGHT
            ball_speed_x = self.ball.speed_x / self.max_ball_speed
            ball_speed_y = self.ball.speed_y / self.max_ball_speed
        except (AttributeError, ZeroDivisionError):
            return np.array([0.5] * self.input_size).reshape(1, -1)

        paddle_y = self.paddle.rect.y / self.paddle.game.HEIGHT
        paddle_center = self.paddle.rect.centery / self.paddle.game.HEIGHT

        paddle_x = self.paddle.rect.centerx / self.paddle.game.WIDTH
        ball_distance = abs(ball_x - paddle_x)

        predicted_y = self._predict_ball_y()
        predicted_y_norm = (
            predicted_y / self.paddle.game.HEIGHT if predicted_y is not None else 0.5
        )

        powerup_data = [0.0, 0.0, 0.0, 0.0]
        if (
            hasattr(self.paddle.game, "powerup_manager")
            and self.paddle.game.powerup_manager
        ):
            pm = self.paddle.game.powerup_manager
            ai_data = pm.get_ai_state_data()

            if self.is_left_paddle:
                powerup_data = [
                    1.0 if ai_data["player_size_boost"] else 0.0,
                    1.0 if ai_data["player_speed_boost"] else 0.0,
                    ai_data["ball_speed_ratio"] - 1.0,
                    ai_data["combo_multiplier"] - 1.0,
                ]
            else:
                powerup_data = [
                    1.0 if ai_data["opponent_size_boost"] else 0.0,
                    1.0 if ai_data["opponent_speed_boost"] else 0.0,
                    ai_data["ball_speed_ratio"] - 1.0,
                    ai_data["combo_multiplier"] - 1.0,
                ]

        state = np.array(
            [
                ball_x,
                ball_y,
                ball_speed_x,
                ball_speed_y,
                paddle_y,
                paddle_center,
                ball_distance,
                predicted_y_norm,
                powerup_data[0],  # Paddle size bonus
                powerup_data[1],  # Paddle speed bonus
                powerup_data[2],  # Ball speed modifier
                powerup_data[3],  # Combo multiplier
            ]
        )
        return state.reshape(1, -1)

    def _predict_ball_y(self):
        """
        Human-like ball prediction with realistic limitations.

        Returns:
            float: Estimated ball position or None if ball not approaching
        """
        if not self.ball or not hasattr(self.ball, "speed_x"):
            return None

        if self.ball.speed_x == 0:
            return None

        moving_towards = (self.is_left_paddle and self.ball.speed_x < 0) or (
            not self.is_left_paddle and self.ball.speed_x > 0
        )

        if not moving_towards:
            return None

        distance_to_paddle = abs(self.ball.rect.centerx - self.paddle.rect.centerx)
        time_to_paddle = distance_to_paddle / abs(self.ball.speed_x)

        if time_to_paddle > self.prediction_time_limit:
            time_to_paddle = self.prediction_time_limit

        predicted_y = self.ball.rect.centery + self.ball.speed_y * time_to_paddle

        distance_factor = distance_to_paddle / self.paddle.game.WIDTH
        error_amount = distance_factor * self.distance_error_factor
        prediction_error = random.uniform(-error_amount, error_amount)
        predicted_y += prediction_error

        if predicted_y < 0:
            predicted_y = abs(predicted_y) * self.wall_bounce_accuracy
        elif predicted_y > self.paddle.game.HEIGHT:
            over_amount = predicted_y - self.paddle.game.HEIGHT
            predicted_y = (
                self.paddle.game.HEIGHT - over_amount * self.wall_bounce_accuracy
            )

        error_range = self.base_prediction_error
        predicted_y += random.uniform(-error_range, error_range)

        predicted_y = max(0, min(self.paddle.game.HEIGHT, predicted_y))

        return predicted_y

    def get_action(self, state):
        """
        Get action from neural network with improved logic.

        Args:
            state: Current game state

        Returns:
            int: Movement direction (-1: up, 0: stop, 1: down)
        """

        prediction = self.neural_network.forward(state)[0][0] + random.uniform(
            -0.05, 0.05
        )

        if prediction < 0.4:
            action = -1
        elif prediction > 0.6:
            action = 1
        else:
            if random.random() < 0.3:
                action = 0
            else:
                predicted_y = state[0][7]
                paddle_y = state[0][4]
                if predicted_y < paddle_y:
                    action = -1
                else:
                    action = 1

        return action

    def update(self):
        """Update AI state and make movement decision"""
        try:
            state = self.get_state()
            action = self.get_action(state)
            self.paddle.move(action)

            self.last_paddle_y = self.paddle.rect.y
        except Exception:
            try:
                center_y = self.paddle.game.HEIGHT // 2
                if self.paddle.rect.centery < center_y - 10:
                    self.paddle.move(1)
                elif self.paddle.rect.centery > center_y + 10:
                    self.paddle.move(-1)
            except Exception:
                pass

    def reset_round_metrics(self):
        """Reset metrics for the current round"""
        if self.round_total_hits > 0:
            self.update_long_term_metrics()

        self.round_successful_hits = 0
        self.round_total_hits = 0
        self.round_consecutive_hits = 0
        self.round_consecutive_misses = 0
        self.round_rewards = 0

        self.ball_in_paddle_zone = False
        self.opportunity_processed = False
        self.last_ball_x = self.ball.rect.x
        self.last_ball_speed_x = self.ball.speed_x

    def update_long_term_metrics(self):
        """Update long-term metrics based on current round performance"""
        self.long_term_successful_hits += self.round_successful_hits
        self.long_term_total_hits += self.round_total_hits
        self.long_term_rewards += self.round_rewards
        self.long_term_episodes += 1

        if self.round_consecutive_hits > self.long_term_consecutive_hits:
            self.long_term_consecutive_hits = self.round_consecutive_hits

        if self.round_consecutive_hits > self.long_term_best_consecutive_hits:
            self.long_term_best_consecutive_hits = self.round_consecutive_hits

        round_accuracy = self.round_successful_hits / max(1, self.round_total_hits)
        self.performance_history.append(round_accuracy)
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)

        self._update_difficulty()

    def _update_difficulty(self):
        """Update difficulty based on recent performance"""
        if len(self.performance_history) < 3:
            return

        recent_performance = sum(self.performance_history[-3:]) / 3
        if recent_performance > 0.8:
            self.current_difficulty *= 1.2
        elif recent_performance < 0.3:
            self.current_difficulty *= 0.8

        self.current_difficulty = max(0.5, min(2.0, self.current_difficulty))
        self.difficulty_history.append(self.current_difficulty)

    def get_performance_metrics(self):
        """Get comprehensive performance metrics"""
        try:
            round_accuracy = self.round_successful_hits / max(1, self.round_total_hits)
            round_accuracy = max(0.0, min(1.0, round_accuracy))

            long_term_accuracy = self.long_term_successful_hits / max(
                1, self.long_term_total_hits
            )
            long_term_accuracy = max(0.0, min(1.0, long_term_accuracy))

            recent_performance = 0.5
            if self.performance_history:
                recent_performance = sum(self.performance_history[-10:]) / min(
                    10, len(self.performance_history)
                )
                recent_performance = max(0.0, min(1.0, recent_performance))

            return {
                "round_accuracy": round_accuracy,
                "long_term_accuracy": long_term_accuracy,
                "recent_performance": recent_performance,
                "best_consecutive_hits": max(
                    self.long_term_consecutive_hits, self.round_consecutive_hits
                ),
                "current_difficulty": max(0.1, min(10.0, self.current_difficulty)),
                "total_episodes": max(0, self.long_term_episodes),
                "total_rewards": self.long_term_rewards,
            }
        except Exception:
            return {
                "round_accuracy": 0.0,
                "long_term_accuracy": 0.0,
                "recent_performance": 0.0,
                "best_consecutive_hits": 0,
                "current_difficulty": 1.0,
                "total_episodes": 0,
                "total_rewards": 0,
            }

    def calculate_reward(self):
        """
        Calculate reward for training with proper metrics tracking.

        Returns:
            float: Reward value
        """
        reward = 0

        current_ball_x = self.ball.rect.x
        current_ball_speed_x = self.ball.speed_x

        ball_approaching = False
        if self.is_left_paddle:
            ball_approaching = (
                current_ball_speed_x < 0
                and current_ball_x > self.paddle.rect.right + 10
            )
        else:
            ball_approaching = (
                current_ball_speed_x > 0 and current_ball_x < self.paddle.rect.left - 10
            )

        collision_detected = False

        if self.ball.rect.colliderect(self.paddle.rect):
            collision_detected = True

        elif ball_approaching:
            ball_vertically_aligned = (
                self.ball.rect.centery >= self.paddle.rect.top
                and self.ball.rect.centery <= self.paddle.rect.bottom
            )

            if ball_vertically_aligned:
                if self.is_left_paddle:
                    collision_detected = (
                        self.ball.rect.left <= self.paddle.rect.right
                        and self.ball.rect.right >= self.paddle.rect.left
                    )
                else:
                    collision_detected = (
                        self.ball.rect.right >= self.paddle.rect.left
                        and self.ball.rect.left <= self.paddle.rect.right
                    )

        if ball_approaching and not self.ball_in_paddle_zone:
            self.ball_in_paddle_zone = True
            self.opportunity_processed = False

        if self.ball_in_paddle_zone and not ball_approaching:
            if not self.opportunity_processed:
                if collision_detected or self._was_ball_hit_recently():
                    self.successful_hits += 1
                    self.round_successful_hits += 1
                    self.consecutive_hits += 1
                    self.round_consecutive_hits += 1
                    self.consecutive_misses = 0
                    self.round_consecutive_misses = 0
                    reward += 2.0 * self.current_difficulty

                else:
                    self.consecutive_misses += 1
                    self.round_consecutive_misses += 1
                    self.consecutive_hits = 0
                    self.round_consecutive_hits = 0
                    reward -= 3.0 / self.current_difficulty

                self.total_hits += 1
                self.round_total_hits += 1
                self.opportunity_processed = True

            self.ball_in_paddle_zone = False

        if self.last_paddle_y is not None:
            if self.last_paddle_y == self.paddle.rect.y:
                reward -= 0.1
            else:
                movement = abs(self.paddle.rect.y - self.last_paddle_y)
                reward += 0.1 * (movement / self.paddle.speed)

        if ball_approaching and self.ball.speed_x != 0:
            predicted_y = self._predict_ball_y()
            if predicted_y is not None:
                paddle_center = self.paddle.rect.y + self.paddle.rect.height / 2
                distance_to_predicted = abs(paddle_center - predicted_y)
                if self.paddle.game.HEIGHT > 0:
                    prediction_accuracy = 1.0 - min(
                        distance_to_predicted / self.paddle.game.HEIGHT, 1.0
                    )
                    reward += 0.2 * prediction_accuracy

        if ball_approaching and not self.paddle.moving:
            reward -= 0.05

        self.last_ball_x = current_ball_x
        self.last_ball_speed_x = current_ball_speed_x

        self.round_rewards += reward
        return reward

    def _was_ball_hit_recently(self):
        """
        Check if ball was hit recently based on collision cooldown.

        Returns:
            bool: True if ball was likely hit recently
        """
        if hasattr(self.ball, "last_collision_time") and hasattr(
            self.ball, "collision_cooldown"
        ):

            current_time = time.time()
            time_since_collision = current_time - self.ball.last_collision_time

            cooldown_window = self.ball.collision_cooldown * 5
            was_recent = time_since_collision < cooldown_window

            return was_recent
        return False

    def save_model(self, filepath):
        """
        Save trained model.

        Args:
            filepath: Path to save file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.neural_network.save_model(filepath)

    def _save_model_with_accuracy(self, accuracy):
        """
        Save model with accuracy in filename.

        Args:
            accuracy: Current model accuracy
        """
        accuracy_str = f"{int(accuracy * 100)}"

        model_path = os.path.join("ai_models", f"model_{accuracy_str}.pkl")
        state_path = os.path.join("ai_models", f"state_{accuracy_str}.pkl")

        self.save_model(model_path)
        self.save_state(state_path)

    def train(self):
        """Train neural network"""
        try:
            state = self.get_state()
            current_action = self.get_action(state)
            self.calculate_reward()

            target = np.array([[0.5]])
            if current_action == -1:
                target = np.array([[0.0]])
            elif current_action == 1:
                target = np.array([[1.0]])

            self.neural_network.train_batch(state, target, self.learning_rate)
            self.episodes += 1

            round_accuracy = self.round_successful_hits / max(1, self.round_total_hits)
            if round_accuracy < 0.3:
                self.learning_rate = self.base_learning_rate * 1.5
            elif round_accuracy > 0.7:
                self.learning_rate = self.base_learning_rate * 0.8
            else:
                self.learning_rate = self.base_learning_rate

            if self.episodes % 100 == 0:
                total_accuracy = self.successful_hits / max(1, self.total_hits)

                for threshold in self.accuracy_thresholds:
                    if (
                        total_accuracy >= threshold
                        and threshold not in self.saved_thresholds
                    ):
                        self._save_model_with_accuracy(total_accuracy)
                        self.saved_thresholds.add(threshold)

            if self.episodes % self.save_interval == 0:
                self.save_model("ai_models/current_model.pkl")

        except Exception:
            pass

    def load_model(self, filepath):
        """
        Load trained model.

        Args:
            filepath: Path to saved model file
        """
        self.neural_network.load_model(filepath)

    def save_state(self, filepath):
        """
        Save AI controller state.

        Args:
            filepath: Path to save state file
        """
        state_data = {
            "last_state": self.get_state(),
            "last_action": self.get_action(self.get_state()),
            "total_rewards": self.total_rewards,
            "episodes": self.episodes,
            "successful_hits": self.successful_hits,
            "total_hits": self.total_hits,
            "max_ball_speed": self.max_ball_speed,
            "consecutive_misses": self.consecutive_misses,
            "consecutive_hits": self.consecutive_hits,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state_data, f)

    def load_state(self, filepath):
        """
        Load AI controller state.

        Args:
            filepath: Path to saved state file
        """
        with open(filepath, "rb") as f:
            state_data = pickle.load(f)

        self.last_state = state_data.get("last_state")
        self.last_action = state_data.get("last_action")
        self.total_rewards = state_data.get("total_rewards", 0)
        self.episodes = state_data.get("episodes", 0)
        self.successful_hits = state_data.get("successful_hits", 0)
        self.total_hits = state_data.get("total_hits", 0)
        self.max_ball_speed = state_data.get("max_ball_speed", 0)

        self.consecutive_misses = state_data.get("consecutive_misses", 0)
        self.consecutive_hits = state_data.get("consecutive_hits", 0)

    def reset_training(self):
        """Reset training state"""
        self.episodes = 0
        self.successful_hits = 0
        self.total_hits = 0
        self.total_rewards = 0
        self.consecutive_misses = 0
        self.consecutive_hits = 0
        self.learning_rate = self.base_learning_rate

        self.neural_network = NeuralNetwork(
            self.neural_network.input_size,
            self.neural_network.hidden_size,
            self.neural_network.output_size,
            self.base_learning_rate,
        )

    def start_training(self):
        """Start training the AI controller"""
        self.training = True

    def stop_training(self):
        """Stop training the AI controller"""
        self.training = False

    def validate_hit_detection(self, actual_collision_occurred=False):
        """
        Validate and synchronize hit detection with main game.

        Args:
            actual_collision_occurred (bool): Whether main game detected a collision
        """
        if actual_collision_occurred and self.ball_in_paddle_zone:
            if not self.opportunity_processed:
                self.successful_hits += 1
                self.round_successful_hits += 1
                self.consecutive_hits += 1
                self.round_consecutive_hits += 1
                self.consecutive_misses = 0
                self.round_consecutive_misses = 0
                self.total_hits += 1
                self.round_total_hits += 1
                self.opportunity_processed = True

    def notify_game_event(self, event_type, details=None):
        """
        Notify AI controller of significant game events.

        Args:
            event_type (str): Type of event
            details (dict): Additional event details
        """
        if event_type == "goal_scored":
            self.round_rewards += 5.0 * self.current_difficulty
        elif event_type == "goal_conceded":
            self.round_rewards -= 5.0 / self.current_difficulty
        elif event_type == "round_end":
            if self.round_total_hits > 0:
                self.update_long_term_metrics()
        elif event_type == "paddle_hit":
            self.validate_hit_detection(True)
