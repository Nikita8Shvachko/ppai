import pygame
import numpy as np
from typing import Dict, List, Tuple
import time


class NeuralNetworkVisualizer:

    def __init__(
        self,
        width: int = 250,
        height: int = 300,
        position: Tuple[int, int] = (10, 10),
        screen_width: int = 1200,
        screen_height: int = 900,
    ):
        """
        Initialize the NeuralNetworkVisualizer.
        Args:
            width: The width of the visualizer
            height: The height of the visualizer
            position: The position of the visualizer
            screen_width: The width of the screen
            screen_height: The height of the screen
        Returns:
            None
        """
        self.width = width
        self.height = height
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.position = self._clamp_position_to_screen(position)
        self.enabled = True

        self.cached_activations = None
        self.cached_weights = None
        self.needs_redraw = True
        self.last_update_time = 0
        self.update_frequency = 0.05

        self.background_surface = None
        self.network_surface = None
        self.layer_spacing = 60
        self.max_neurons_per_layer = 12
        self.max_connections_per_layer = 20
        self.neuron_radius = 8

        self.bg_color = (20, 20, 20, 180)
        self.text_color = (200, 200, 200)
        self.neuron_color = (100, 150, 255)
        self.weight_positive_color = (0, 255, 0)
        self.weight_negative_color = (255, 0, 0)
        pygame.font.init()
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)

    def _clamp_position_to_screen(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """
        Clamp the position of the neural network visualizer to the screen.
        Args:
            position: The position to clamp
        Returns:
            Tuple[int, int]: The clamped position
        """
        x, y = position
        original_position = (x, y)

        max_x = self.screen_width - self.width - 10
        x = min(x, max_x)
        x = max(x, 10)

        max_y = self.screen_height - self.height - 10
        y = min(y, max_y)
        y = max(y, 10)

        clamped_position = (x, y)

        if original_position != clamped_position:
            print(
                f"Neural network visualizer position adjusted: {original_position} -> {clamped_position}"
            )
            print(
                f"Screen bounds: {self.screen_width}x{self.screen_height}, "
                f"Visualizer size: {self.width}x{self.height}"
            )

        return clamped_position

    def set_enabled(self, enabled: bool):
        """
        Set the enabled state of the neural network visualizer.
        Args:
            enabled: Whether to enable or disable the visualizer
        Returns:
            None
        """
        self.enabled = enabled
        if enabled:
            self.needs_redraw = True

    def toggle_enabled(self):
        """
        Toggle the enabled state of the neural network visualizer.
        Returns:
            None
        """
        self.set_enabled(not self.enabled)

    def set_position(self, position: Tuple[int, int]):
        """
        Set the position of the neural network visualizer.
        Args:
            position: The position to set
        Returns:
            None
        """
        self.position = self._clamp_position_to_screen(position)
        self.needs_redraw = True

    def update(self, neural_network):
        """
        Update the neural network visualizer.
        Args:
            neural_network: The neural network to update
        Returns:
            None
        """
        if not self.enabled:
            return

        current_time = time.time()
        if current_time - self.last_update_time < self.update_frequency:
            return
        self.last_update_time = current_time

        try:
            if hasattr(neural_network, "inputs"):
                activations = {
                    "input": neural_network.inputs,
                    "hidden1": getattr(neural_network, "hidden1_layer", None),
                    "hidden2": getattr(neural_network, "hidden2_layer", None),
                    "output": getattr(neural_network, "output_layer", None),
                }
            else:
                activations = {}

            weights = {}
            if hasattr(neural_network, "weights_input_hidden1"):
                weights["input_to_hidden1"] = neural_network.weights_input_hidden1
            if hasattr(neural_network, "weights_hidden2_output"):
                weights["hidden2_to_output"] = neural_network.weights_hidden2_output

            if self._should_update(activations, weights):
                self.cached_activations = activations
                self.cached_weights = weights
                self.needs_redraw = True

        except Exception:
            pass

    def _should_update(self, activations: Dict, weights: Dict) -> bool:
        """
        Check if the network should be updated.
        Args:
            activations: The activations of the network
            weights: The weights of the network
        Returns:
            bool: True if the network should be updated, False otherwise
        """
        if self.cached_activations is None:
            return True

        try:
            for key, value in activations.items():
                if value is not None and key in self.cached_activations:
                    cached_value = self.cached_activations[key]
                    if cached_value is not None:
                        if np.mean(np.abs(value - cached_value)) > 0.01:
                            return True
            return False
        except Exception:
            return True

    def draw(self, surface: pygame.Surface):
        """
        Draw the network.
        Args:
            surface: The surface to draw on
        Returns:
            None
        """
        if not self.enabled:
            return

        if self.background_surface is None or self.needs_redraw:
            self._create_background_surface()

        if self.network_surface is None or self.needs_redraw:
            self._create_network_surface()
            self.needs_redraw = False

        surface.blit(self.background_surface, self.position)
        if self.network_surface:
            surface.blit(self.network_surface, self.position)

    def _create_background_surface(self):
        """
        Create the background surface.
        Returns:
            None
        """
        self.background_surface = pygame.Surface(
            (self.width, self.height), pygame.SRCALPHA
        )

        bg_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self.background_surface, self.bg_color, bg_rect)
        pygame.draw.rect(self.background_surface, (60, 60, 60), bg_rect, 1)

        title_text = self.font.render("Neural Network", True, self.text_color)
        self.background_surface.blit(title_text, (10, 10))

        labels = ["Input", "Hidden", "Output"]
        layer_x_positions = self._get_layer_positions()

        for i, (label, x_pos) in enumerate(zip(labels, layer_x_positions)):
            label_text = self.small_font.render(label, True, self.text_color)
            text_rect = label_text.get_rect()
            text_rect.centerx = x_pos
            text_rect.y = 35
            self.background_surface.blit(label_text, text_rect)

    def _create_network_surface(self):
        """
        Create the network surface.
        Returns:
            None
        """
        self.network_surface = pygame.Surface(
            (self.width, self.height), pygame.SRCALPHA
        )

        if self.cached_activations is None:
            return

        layer_x_positions = self._get_layer_positions()

        hidden1 = self.cached_activations.get("hidden1")
        hidden2 = self.cached_activations.get("hidden2")
        combined_hidden = None
        if hidden1 is not None and hidden2 is not None:
            combined_hidden = np.concatenate([hidden1.flatten(), hidden2.flatten()])
            combined_hidden = combined_hidden.reshape(-1, 1)

        layer_data = [
            ("input", self.cached_activations.get("input")),
            ("hidden_combined", combined_hidden),
            ("output", self.cached_activations.get("output")),
        ]

        self._draw_connections(layer_x_positions, layer_data)
        self._draw_neurons(layer_x_positions, layer_data)

    def _get_layer_positions(self) -> List[int]:
        """
        Get the layer positions.
        Returns:
            List[int]: The layer positions
        """
        start_x = 40
        return [start_x + i * self.layer_spacing for i in range(3)]

    def _draw_connections(self, layer_x_positions: List[int], layer_data: List[Tuple]):
        """
        Draw the connections.
        Args:
            layer_x_positions: The x positions of the layers
            layer_data: The data of the layers
        Returns:
            None
        """
        if self.cached_weights is None:
            return

        weight_matrices = [
            self.cached_weights.get("input_to_hidden1"),
            self.cached_weights.get("hidden2_to_output"),
        ]

        for i in range(len(weight_matrices)):
            if weight_matrices[i] is None:
                continue

            from_layer = layer_data[i][1]
            to_layer = layer_data[i + 1][1]

            if from_layer is None or to_layer is None:
                continue

            from_positions = self._get_neuron_positions(
                layer_x_positions[i], from_layer
            )
            to_positions = self._get_neuron_positions(
                layer_x_positions[i + 1], to_layer
            )

            max_connections = 30
            weights = weight_matrices[i]

            if weights.size > max_connections:
                sample_indices = np.random.choice(
                    weights.size, max_connections, replace=False
                )
                for idx in sample_indices:
                    from_idx = idx // weights.shape[1]
                    to_idx = idx % weights.shape[1]

                    if from_idx < len(from_positions) and to_idx < len(to_positions):
                        weight = weights[from_idx, to_idx]
                        self._draw_connection(
                            from_positions[from_idx], to_positions[to_idx], weight
                        )
            else:
                for from_idx in range(min(len(from_positions), weights.shape[0])):
                    for to_idx in range(min(len(to_positions), weights.shape[1])):
                        weight = weights[from_idx, to_idx]
                        self._draw_connection(
                            from_positions[from_idx], to_positions[to_idx], weight
                        )

    def _draw_connection(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], weight: float
    ):
        """
        Draw a connection.
        Args:
            from_pos: The position of the from neuron
            to_pos: The position of the to neuron
            weight: The weight of the connection
        Returns:
            None
        """
        weight_strength = min(1.0, abs(weight) * 2)

        if weight_strength < 0.1:
            return

        if weight > 0:
            color = self.weight_positive_color
        else:
            color = self.weight_negative_color

        line_thickness = max(1, int(weight_strength * 3))

        alpha = int(weight_strength * 120)

        min_x = min(from_pos[0], to_pos[0]) - line_thickness
        max_x = max(from_pos[0], to_pos[0]) + line_thickness
        min_y = min(from_pos[1], to_pos[1]) - line_thickness
        max_y = max(from_pos[1], to_pos[1]) + line_thickness

        surface_width = max_x - min_x + 1
        surface_height = max_y - min_y + 1

        line_surface = pygame.Surface((surface_width, surface_height), pygame.SRCALPHA)

        rel_from_pos = (from_pos[0] - min_x, from_pos[1] - min_y)
        rel_to_pos = (to_pos[0] - min_x, to_pos[1] - min_y)

        color_with_alpha = (*color, alpha)
        pygame.draw.line(
            line_surface, color_with_alpha, rel_from_pos, rel_to_pos, line_thickness
        )

        self.network_surface.blit(line_surface, (min_x, min_y))

    def _draw_neurons(self, layer_x_positions: List[int], layer_data: List[Tuple]):
        """
        Draw the neurons.
        Args:
            layer_x_positions: The x positions of the layers
            layer_data: The data of the layers
        Returns:
            None
        """
        for i, (layer_name, activations) in enumerate(layer_data):
            if activations is None:
                continue

            positions = self._get_neuron_positions(layer_x_positions[i], activations)

            for j, (pos, activation) in enumerate(
                zip(positions, activations.flatten())
            ):
                self._draw_neuron(pos, activation)

    def _get_neuron_positions(self, x_pos: int, activations) -> List[Tuple[int, int]]:
        """
        Get the neuron positions.
        Args:
            x_pos: The x position of the neuron
            activations: The activations of the neuron
        Returns:
            List[Tuple[int, int]]: The neuron positions
        """
        if activations is None:
            return []

        num_neurons = min(len(activations.flatten()), self.max_neurons_per_layer)
        start_y = 60
        available_height = self.height - 100

        if num_neurons == 1:
            return [(x_pos, start_y + available_height // 2)]

        spacing = available_height // (num_neurons - 1) if num_neurons > 1 else 0
        return [(x_pos, start_y + i * spacing) for i in range(num_neurons)]

    def _draw_neuron(self, position: Tuple[int, int], activation: float):
        """
        Draw a neuron.
        Args:
            position: The position of the neuron
            activation: The activation of the neuron
        Returns:
            None
        """
        normalized_activation = max(0, min(1, abs(activation)))

        if normalized_activation > 0.7:
            color = (0, 255, 255)
        elif normalized_activation > 0.3:
            r = int(100 + normalized_activation * 155)
            g = int(50 + normalized_activation * 205)
            b = 255
            color = (r, g, b)
        else:
            intensity = 50 + int(normalized_activation * 100)
            color = (intensity, intensity, 255)

        if normalized_activation > 0.5:
            glow_radius = self.neuron_radius + 4
            glow_color = (*color, 80)
            glow_surface = pygame.Surface(
                (glow_radius * 2, glow_radius * 2), pygame.SRCALPHA
            )
            pygame.draw.circle(
                glow_surface, glow_color, (glow_radius, glow_radius), glow_radius
            )
            self.network_surface.blit(
                glow_surface, (position[0] - glow_radius, position[1] - glow_radius)
            )

        pygame.draw.circle(self.network_surface, color, position, self.neuron_radius)
        pygame.draw.circle(
            self.network_surface, (255, 255, 255), position, self.neuron_radius, 2
        )

        if normalized_activation > 0.3:
            activation_text = self.small_font.render(
                f"{activation:.2f}", True, (255, 255, 255)
            )
            text_rect = activation_text.get_rect()
            text_rect.center = (position[0], position[1] - self.neuron_radius - 12)
            self.network_surface.blit(activation_text, text_rect)

    def get_input_info(self) -> str:
        """
        Get the input info.
        Returns:
            str: The input info
        """
        return "Press 'V' to toggle neural network visualization"


class NeuralNetworkVisualizerManager:

    def __init__(
        self,
        max_visualizers: int = 2,
        screen_width: int = 1200,
        screen_height: int = 900,
    ):
        """
        Initialize the NeuralNetworkVisualizerManager.
        Args:
            max_visualizers: The maximum number of visualizers
            screen_width: The width of the screen
            screen_height: The height of the screen
        """
        self.max_visualizers = max_visualizers
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.visualizers: Dict[str, NeuralNetworkVisualizer] = {}
        self.active_visualizers: List[str] = []

    def add_visualizer(
        self, name: str, position: Tuple[int, int] = None
    ) -> NeuralNetworkVisualizer:
        """
        Add a visualizer.
        Args:
            name: The name of the visualizer
            position: The position of the visualizer
        Returns:
            NeuralNetworkVisualizer: The visualizer
        """
        if len(self.visualizers) >= self.max_visualizers:
            return None

        if position is None:
            visualizer_width = 250
            spacing = visualizer_width + 20
            position = (10 + len(self.visualizers) * spacing, 150)

        visualizer = NeuralNetworkVisualizer(
            position=position,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
        )
        self.visualizers[name] = visualizer
        self.active_visualizers.append(name)

        return visualizer

    def remove_visualizer(self, name: str):
        """
        Remove a visualizer.
        Args:
            name: The name of the visualizer
        Returns:
            None
        """
        if name in self.visualizers:
            del self.visualizers[name]
            if name in self.active_visualizers:
                self.active_visualizers.remove(name)

    def update(self, neural_networks: Dict[str, object]):
        """
        Update the visualizers.
        Args:
            neural_networks: The neural networks to update
        Returns:
            None
        """
        for name, visualizer in self.visualizers.items():
            if name in neural_networks:
                visualizer.update(neural_networks[name])

    def draw(self, surface: pygame.Surface):
        """
        Draw the visualizers.
        Args:
            surface: The surface to draw on
        Returns:
            None
        """
        for name in self.active_visualizers:
            if name in self.visualizers:
                self.visualizers[name].draw(surface)

    def toggle_all(self):
        """
        Toggle the enabled state for all visualizers.
        Returns:
            None
        """
        for visualizer in self.visualizers.values():
            visualizer.toggle_enabled()

    def set_all_enabled(self, enabled: bool):
        """
        Set the enabled state for all visualizers.
        Args:
            enabled: Whether to enable or disable the visualizers
        Returns:
            None
        """
        for visualizer in self.visualizers.values():
            visualizer.set_enabled(enabled)
