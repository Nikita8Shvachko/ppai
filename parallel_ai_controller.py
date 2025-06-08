import os
import queue
import threading
import time
import random
from typing import List, Tuple

import numpy as np

from ai_controller import AIController


class ParallelAIController:

    def __init__(
        self,
        paddle,
        ball,
        num_agents=16,
        input_size=12,
        hidden_size=32,
        output_size=1,
        exchange_interval=500,
        mutation_rate=0.1,
        batch_size=32,
        elite_size=4,
        difficulty_preset=None,
    ):
        """
        Initialize the ParallelAIController.
        Args:
            paddle (Paddle): The paddle to control
            ball (Ball): The ball to track
            num_agents (int): The number of agents to create
            input_size (int): The size of the input layer
            hidden_size (int): The size of the hidden layer
            output_size (int): The size of the output layer
            exchange_interval (int): The number of episodes between experience exchanges
            mutation_rate (float): The rate at which weights are mutated
            batch_size (int): The number of episodes to train on at once
            elite_size (int): The number of agents to select for experience exchange
            difficulty_preset (dict): The difficulty preset to apply to the agents
        Returns:
            None
        """

        self.paddle = paddle
        self.ball = ball
        self.num_agents = num_agents
        self.exchange_interval = exchange_interval
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size
        self.elite_size = elite_size
        self.agents: List[AIController] = []
        self.best_agent = None
        self.best_accuracy = 0.0
        self.exchange_queue = queue.Queue()
        self.training = False
        self.threads = []
        self.agent_performances = {}
        self.best_agent_id = None
        self.best_long_term_accuracy = 0.0
        self.training_batches = []
        self.thread_lock = threading.Lock()
        self.agent_scores = [0] * num_agents
        self.current_agent_index = 0
        self.episodes = 0
        self.last_exchange_time = time.time()
        self.min_exchange_interval = 1.0

        self.difficulty_preset = difficulty_preset

        self.round_successful_hits = 0
        self.round_total_hits = 0
        self.round_consecutive_hits = 0
        self.round_consecutive_misses = 0
        self.round_rewards = 0
        for i in range(num_agents):
            agent = AIController(
                paddle=paddle,
                ball=ball,
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
            )
            agent.agent_id = i
            agent.learning_rate = 0.001 * (1 + random.uniform(-0.3, 0.3))
            if difficulty_preset:
                self._apply_difficulty_to_agent(agent, difficulty_preset)

            self.agents.append(agent)
            self.agent_performances[i] = {
                "recent_performance": 0.5,
                "long_term_accuracy": 0.5,
                "total_episodes": 0,
                "best_consecutive_hits": 0,
                "best_consecutive_misses": 0,
            }
            self.training_batches.append([])

        self.models_dir = "ai_models"
        os.makedirs(self.models_dir, exist_ok=True)

        self.best_agent = self.agents[0]

    def _apply_difficulty_to_agent(self, agent, difficulty_preset):
        """
        Apply a difficulty preset to an agent.
        Args:
            agent (AIController): The agent to apply the difficulty to
            difficulty_preset (dict): The difficulty preset to apply
        Returns:
            None
        """
        try:

            prediction_time_limit = difficulty_preset.get("prediction_time_limit", 1.0)
            distance_error_factor = difficulty_preset.get("distance_error_factor", 0)
            base_prediction_error = difficulty_preset.get("base_prediction_error", 0)
            wall_bounce_accuracy = difficulty_preset.get("wall_bounce_accuracy", 1.0)

            agent.prediction_time_limit = prediction_time_limit
            agent.distance_error_factor = distance_error_factor
            agent.base_prediction_error = base_prediction_error
            agent.wall_bounce_accuracy = wall_bounce_accuracy
            agent.difficulty_name = difficulty_preset.get("name", "Unknown")

            difficulty_learning_modifiers = {
                "Beginner": 0.5,
                "Casual Player": 0.7,
                "Skilled Player": 1.0,
                "Expert Player": 1.2,
                "Unbeatable (Original)": 1.5,
            }

            difficulty_name = difficulty_preset.get("name", "Skilled Player")
            learning_modifier = difficulty_learning_modifiers.get(difficulty_name, 1.0)
            agent.learning_rate *= learning_modifier

            agent_id = agent.agent_id
            print(f"Applied difficulty '{difficulty_name}' to agent {agent_id}")

        except Exception as e:
            print(f"Error applying difficulty to agent: {e}")

    def apply_difficulty_preset(self, difficulty_preset):
        """
        Apply a difficulty preset to all agents.
        Args:
            difficulty_preset (dict): The difficulty preset to apply
        Returns:
            None
        """

        try:
            self.difficulty_preset = difficulty_preset

            for agent in self.agents:
                self._apply_difficulty_to_agent(agent, difficulty_preset)

            preset_name = difficulty_preset.get("name", "Unknown")
            num_agents = len(self.agents)
            print(
                f"Applied difficulty preset '{preset_name}' to all "
                f"{num_agents} agents"
            )

        except Exception as e:
            print(f"Error applying difficulty preset: {e}")

    def _mutate_weights(self, agent, mentor):
        """
        Mutate the weights of an agent.
        Args:
            agent (AIController): The agent to mutate
            mentor (AIController): The mentor to copy weights from
        Returns:
            None
        """
        try:

            agent.neural_network.weights_input_hidden1 = (
                mentor.neural_network.weights_input_hidden1.copy()
            )
            agent.neural_network.weights_hidden1_hidden2 = (
                mentor.neural_network.weights_hidden1_hidden2.copy()
            )
            agent.neural_network.weights_hidden2_output = (
                mentor.neural_network.weights_hidden2_output.copy()
            )

            agent.neural_network.bias_hidden1 = (
                mentor.neural_network.bias_hidden1.copy()
            )
            agent.neural_network.bias_hidden2 = (
                mentor.neural_network.bias_hidden2.copy()
            )
            agent.neural_network.bias_output = mentor.neural_network.bias_output.copy()

            agent.neural_network.bn_gamma1 = mentor.neural_network.bn_gamma1.copy()
            agent.neural_network.bn_beta1 = mentor.neural_network.bn_beta1.copy()
            agent.neural_network.bn_gamma2 = mentor.neural_network.bn_gamma2.copy()
            agent.neural_network.bn_beta2 = mentor.neural_network.bn_beta2.copy()
            agent.neural_network.bn_gamma3 = mentor.neural_network.bn_gamma3.copy()
            agent.neural_network.bn_beta3 = mentor.neural_network.bn_beta3.copy()

            mask = (
                np.random.random(agent.neural_network.weights_input_hidden1.shape)
                < self.mutation_rate
            )
            mutations = np.random.normal(
                0, 0.05, agent.neural_network.weights_input_hidden1.shape
            )
            agent.neural_network.weights_input_hidden1 += mask * mutations

            mask = (
                np.random.random(agent.neural_network.weights_hidden1_hidden2.shape)
                < self.mutation_rate
            )
            mutations = np.random.normal(
                0, 0.05, agent.neural_network.weights_hidden1_hidden2.shape
            )
            agent.neural_network.weights_hidden1_hidden2 += mask * mutations

            mask = (
                np.random.random(agent.neural_network.weights_hidden2_output.shape)
                < self.mutation_rate
            )
            mutations = np.random.normal(
                0, 0.05, agent.neural_network.weights_hidden2_output.shape
            )
            agent.neural_network.weights_hidden2_output += mask * mutations

            mask = (
                np.random.random(agent.neural_network.bias_hidden1.shape)
                < self.mutation_rate
            )
            mutations = np.random.normal(
                0, 0.05, agent.neural_network.bias_hidden1.shape
            )
            agent.neural_network.bias_hidden1 += mask * mutations

            mask = (
                np.random.random(agent.neural_network.bias_hidden2.shape)
                < self.mutation_rate
            )
            mutations = np.random.normal(
                0, 0.05, agent.neural_network.bias_hidden2.shape
            )
            agent.neural_network.bias_hidden2 += mask * mutations

            mask = (
                np.random.random(agent.neural_network.bias_output.shape)
                < self.mutation_rate
            )
            mutations = np.random.normal(
                0, 0.05, agent.neural_network.bias_output.shape
            )
            agent.neural_network.bias_output += mask * mutations

        except Exception as e:
            print(f"Error during weight mutation: {e}")

    def _select_next_agent(self):
        """
        Select the next agent to train.
        Returns:
            AIController: The next agent to train
        """
        with self.thread_lock:
            self.current_agent_index = (self.current_agent_index + 1) % self.num_agents
            return self.agents[self.current_agent_index]

    def _update_agent_scores(self, winner_id, loser_id):
        """
        Update the scores of the agents.
        Args:
            winner_id (int): The ID of the winning agent
            loser_id (int): The ID of the losing agent
        Returns:
            None
        """
        with self.thread_lock:
            self.agent_scores[winner_id] += 1

            if max(self.agent_scores) > 1000:
                self.agent_scores = [score // 2 for score in self.agent_scores]

    def _select_agents_for_game(self):
        """
        Select the agents for a game.
        Returns:
            tuple: A tuple containing the top agent and an opponent
        """
        with self.thread_lock:

            sorted_indices = sorted(
                range(len(self.agent_scores)),
                key=lambda i: self.agent_scores[i],
                reverse=True,
            )

            top_agent = self.agents[sorted_indices[0]]
            opponent = self.agents[random.choice(sorted_indices[1:])]
            return top_agent, opponent

    def _train_agent(self, agent_id):
        """
        Train an agent.
        Args:
            agent_id (int): The ID of the agent to train
        Returns:
            None
        """
        try:
            agent = self.agents[agent_id]
            if not agent.training:
                return

            metrics = agent.get_performance_metrics()
            self.agent_performances[agent_id] = metrics

            if metrics["long_term_accuracy"] > self.best_long_term_accuracy:
                self.best_long_term_accuracy = metrics["long_term_accuracy"]
                self.best_agent_id = agent_id
                self.best_agent = agent

            current_time = time.time()
            if (
                current_time - self.last_exchange_time >= self.min_exchange_interval
                and agent.long_term_episodes % self.exchange_interval == 0
            ):
                self._exchange_experience()
                self.last_exchange_time = current_time

        except Exception as e:
            print(f"Error training agent {agent_id}: {e}")

    def _exchange_experience(self):
        """
        Exchange experience between agents.
        Returns:
            None
        """
        try:

            sorted_agents = sorted(
                self.agent_performances.items(),
                key=lambda x: x[1]["recent_performance"],
                reverse=True,
            )

            elite_size = max(2, len(self.agents) // 8)
            elite_agents = [agent_id for agent_id, _ in sorted_agents[:elite_size]]

            for agent_id, agent in enumerate(self.agents):
                if agent_id not in elite_agents:

                    mentor_id = random.choice(elite_agents)
                    mentor = self.agents[mentor_id]

                    self._mutate_weights(agent, mentor)

                    performance = self.agent_performances[agent_id]
                    if performance["recent_performance"] < 0.3:
                        agent.learning_rate *= 1.1
                    elif performance["recent_performance"] > 0.7:
                        agent.learning_rate *= 0.9

        except Exception as e:
            print(f"Error during experience exchange: {e}")

    def _save_best_agent(self):
        """
        Save the best agent.
        Returns:
            None
        """
        if self.best_agent:
            model_path = os.path.join(self.models_dir, "best_model.pkl")
            state_path = os.path.join(self.models_dir, "best_state.pkl")
            try:
                self.best_agent.save_model(model_path)
                self.best_agent.save_state(state_path)
                print(f"Saved best agent with accuracy {self.best_accuracy:.2%}")
            except Exception as e:
                print(f"Error saving agent: {e}")

    def get_agent_statistics(self) -> List[Tuple[int, float]]:
        """
        Get the statistics of the agents.
        Returns:
            list: A list of tuples containing the agent ID and its long-term accuracy
        """
        with self.thread_lock:
            return [
                (i, self.agent_performances[i]["long_term_accuracy"])
                for i in range(len(self.agents))
            ]

    def start_training(self):
        """
        Start training the agents.
        Returns:
            None
        """
        self.training = True
        print("Threading disabled for macOS GIL compatibility")

    def stop_training(self):
        """
        Stop training the agents.
        Returns:
            None
        """
        self.training = False

        for thread in self.threads:
            try:
                thread.join(timeout=1.0)
                if thread.is_alive():
                    print("Warning: Training thread did not stop cleanly")
            except Exception as e:
                print(f"Error stopping training thread: {e}")

        self.threads.clear()
        print("AI training stopped")

    def load_best_agent(self):
        """
        Load the best agent.
        Returns:
            bool: True if the best agent was loaded, False otherwise
        """
        model_path = os.path.join(self.models_dir, "best_model.pkl")
        state_path = os.path.join(self.models_dir, "best_state.pkl")
        try:
            if self.best_agent:
                self.best_agent.load_model(model_path)
                self.best_agent.load_state(state_path)
                print("Loaded best agent")
                return True
            elif self.agents:
                self.agents[0].load_model(model_path)
                self.agents[0].load_state(state_path)
                self.best_agent = self.agents[0]
                self.best_agent_id = 0
                self.best_long_term_accuracy = self.agent_performances[0][
                    "long_term_accuracy"
                ]
                print("Loaded best agent into first agent")
                return True
        except (FileNotFoundError, Exception) as e:
            print(f"Failed to load best agent: {e}")
            return False
        return False

    def update(self):
        """
        Update the agents.
        Returns:
            None
        """
        if self.best_agent:
            self.best_agent.update()
        elif self.agents:

            self.agents[0].update()

    def train(self):
        """
        Train the agents.
        Returns:
            None
        """
        if not self.training:
            return

        active_agent = self.get_active_agent()
        if not active_agent:
            return

        active_agent.train()

        if hasattr(active_agent, "round_successful_hits"):
            self.round_successful_hits = active_agent.round_successful_hits
            self.round_total_hits = active_agent.round_total_hits
            self.round_consecutive_hits = active_agent.round_consecutive_hits
            self.round_consecutive_misses = active_agent.round_consecutive_misses
            self.round_rewards = active_agent.round_rewards

        self.episodes += 1

        if (
            self.episodes - self.last_exchange_time >= self.min_exchange_interval
            and active_agent.long_term_episodes % self.exchange_interval == 0
        ):
            self._exchange_experience()
            self.last_exchange_time = time.time()

        if self.episodes % 50 == 0:
            accuracy = self.round_successful_hits / max(1, self.round_total_hits)
            if accuracy > self.best_accuracy * 1.01:
                self.best_accuracy = accuracy
                self.best_agent = active_agent

    def get_active_agent(self):
        """
        Get the active agent.
        Returns:
            AIController: The active agent
        """

        if self.best_agent:
            return self.best_agent
        elif self.agents:
            return self.agents[0]
        return None

    def reset_training(self):
        """
        Reset the training.
        Returns:
            None
        """
        self.stop_training()
        for agent in self.agents:
            agent.reset_training()
        self.best_agent = None
        self.best_accuracy = 0.0
        self.agent_performances = {}
        for batch in self.training_batches:
            batch.clear()
        self.start_training()
        print("Reset training for all agents")

    def load_model(self, filepath):
        """
        Load a model into the active agent.
        Args:
            filepath (str): The path to the model file
        Returns:
            None
        """
        active_agent = self.get_active_agent()
        if active_agent:
            active_agent.load_model(filepath)
            print("Model loaded into active agent")
        else:
            print("No active agent to load model into")

    def save_model(self, filepath):
        """
        Save the model from the active agent.
        Args:
            filepath (str): The path to the model file
        Returns:
            None
        """
        active_agent = self.get_active_agent()
        if active_agent:
            active_agent.save_model(filepath)
            print("Active agent model saved")
        else:
            print("No active agent to save model from")

    def get_statistics_summary(self):
        """
        Get the statistics summary.
        Returns:
            str: The statistics summary
        """
        try:
            if not self.agents:
                return "No agents available"

            total_accuracy = 0
            total_episodes = 0
            best_consecutive = 0
            active_agents = 0

            for agent_id, metrics in self.agent_performances.items():
                if metrics["total_episodes"] > 0:
                    total_accuracy += metrics["long_term_accuracy"]
                    total_episodes += metrics["total_episodes"]
                    best_consecutive = max(
                        best_consecutive, metrics["best_consecutive_hits"]
                    )
                    active_agents += 1

            if active_agents == 0:
                return "No active agents"

            avg_accuracy = total_accuracy / active_agents
            avg_episodes = total_episodes / active_agents

            return (
                f"Agents: {len(self.agents)} active, {active_agents} training\n"
                f"Accuracy: {avg_accuracy:.1%} (best: {self.best_long_term_accuracy:.1%})\n"
                f"Episodes: {avg_episodes:.0f} per agent\n"
                f"Best streak: {best_consecutive} hits\n"
                f"Learning rate: {self.agents[0].learning_rate:.3f}"
            )

        except Exception as e:
            return f"Error getting statistics: {e}"

    def reset_round_metrics(self):
        """
        Reset the round metrics.
        Returns:
            None
        """
        for agent in self.agents:
            has_hits = hasattr(agent, "round_total_hits")
            if has_hits and agent.round_total_hits > 0:
                if hasattr(agent, "update_long_term_metrics"):
                    agent.update_long_term_metrics()

        self.round_successful_hits = 0
        self.round_total_hits = 0
        self.round_consecutive_hits = 0
        self.round_consecutive_misses = 0
        self.round_rewards = 0

        for agent in self.agents:
            if hasattr(agent, "reset_round_metrics"):
                agent.reset_round_metrics()

    def get_performance_metrics(self):
        """
        Get the performance metrics.
        Returns:
            dict: The performance metrics
        """
        try:

            active_agent = self.get_active_agent()
            if not active_agent:
                return {
                    "round_accuracy": 0.0,
                    "long_term_accuracy": 0.0,
                    "recent_performance": 0.0,
                    "best_consecutive_hits": 0,
                    "current_difficulty": 1.0,
                    "total_episodes": 0,
                    "total_rewards": 0,
                }

            metrics = active_agent.get_performance_metrics()

            if metrics:
                metrics["round_accuracy"] = max(
                    0.0, min(1.0, metrics.get("round_accuracy", 0.0))
                )
                metrics["long_term_accuracy"] = max(
                    0.0, min(1.0, metrics.get("long_term_accuracy", 0.0))
                )
                metrics["recent_performance"] = max(
                    0.0, min(1.0, metrics.get("recent_performance", 0.0))
                )
                metrics["best_consecutive_hits"] = max(
                    0, metrics.get("best_consecutive_hits", 0)
                )
                metrics["current_difficulty"] = max(
                    0.1, min(10.0, metrics.get("current_difficulty", 1.0))
                )
                metrics["total_episodes"] = max(0, metrics.get("total_episodes", 0))

            agent_id = active_agent.agent_id
            self.agent_performances[agent_id] = metrics

            if metrics["long_term_accuracy"] > self.best_long_term_accuracy:
                self.best_long_term_accuracy = metrics["long_term_accuracy"]
                self.best_agent_id = agent_id
                self.best_agent = active_agent

            return metrics

        except Exception as e:
            print(f"Error getting performance metrics: {e}")
            return {
                "round_accuracy": 0.0,
                "long_term_accuracy": 0.0,
                "recent_performance": 0.0,
                "best_consecutive_hits": 0,
                "current_difficulty": 1.0,
                "total_episodes": 0,
                "total_rewards": 0,
            }

    def notify_game_event(self, event_type, details=None):
        """
        Notify the active agent of a game event.
        Args:
            event_type (str): The type of event
            details (dict): The details of the event
        Returns:
            None
        """
        active_agent = self.get_active_agent()
        if active_agent and hasattr(active_agent, "notify_game_event"):
            active_agent.notify_game_event(event_type, details)

    def validate_hit_detection(self, actual_collision_occurred=False):
        """
        Validate the hit detection.
        Args:
            actual_collision_occurred (bool): Whether the collision occurred
        Returns:
            None
        """
        active_agent = self.get_active_agent()
        if active_agent and hasattr(active_agent, "validate_hit_detection"):
            active_agent.validate_hit_detection(actual_collision_occurred)
