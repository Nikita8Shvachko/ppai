import math
import queue
import random
import threading
import time

import pygame

from neural_network_visualizer import NeuralNetworkVisualizerManager
from parallel_ai_controller import ParallelAIController


class NeonTheme:
    """
    NeonTheme class for the Pong game.
    """

    def __init__(self):
        """
        Initialize the NeonTheme class.
        """
        self.NEON_CYAN = (0, 255, 255)
        self.NEON_PINK = (255, 20, 147)
        self.NEON_GREEN = (57, 255, 20)
        self.NEON_PURPLE = (138, 43, 226)
        self.NEON_ORANGE = (255, 165, 0)
        self.NEON_BLUE = (30, 144, 255)
        self.NEON_YELLOW = (255, 255, 0)
        self.DARK_BG = (10, 10, 20)
        self.DARKER_BG = (5, 5, 15)
        self.GLOW_CYAN = (0, 128, 128)
        self.GLOW_PINK = (128, 10, 73)
        self.GLOW_GREEN = (28, 128, 10)
        self.GLOW_PURPLE = (69, 21, 113)
        self.glow_pulse = 0
        self.time_offset = 0
        self.last_update_time = time.time()
        self.ball_particles = []
        self.paddle_particles = []
        self.max_ball_particles = 30
        self.max_paddle_particles = 30
        self.particle_update_counter = 0
        self.skip_particle_frames = 1
        self.ball_particle_interval = 4.5
        self.ball_particle_timer = 0.0
        self.paddle_particle_interval = 4.5
        self.paddle_particle_timer = 0.0

    def update_animations(self, dt):
        """
        Update the animations.
        Args:
            dt (float): The time delta
        Returns:
            None
        """
        dt = min(dt, 0.05)
        if hasattr(self, "game") and hasattr(self.game, "simulation_speed"):
            dt *= min(self.game.simulation_speed, 2.0)
        self.time_offset += dt
        self.glow_pulse = (math.sin(self.time_offset * 3) + 1) * 0.5
        self.particle_update_counter += 1
        if self.particle_update_counter >= self.skip_particle_frames:
            self._update_particles(dt)
            self.particle_update_counter = 0

    def _update_particles(self, dt):
        """
        Update the particles.
        Args:
            dt (float): The time delta
        Returns:
            None
        """
        dt = max(0.0, min(dt, 0.1))
        new_ball_particles = []
        for particle in self.ball_particles:
            new_life = max(0.0, particle["life"] - dt * 1.5)
            if new_life > 0:
                new_alpha = max(0, min(255, new_life * 255))
                new_ball_particles.append(
                    {
                        **particle,
                        "life": new_life,
                        "alpha": new_alpha,
                    }
                )
        self.ball_particles = new_ball_particles
        new_paddle_particles = []
        for particle in self.paddle_particles:
            new_life = max(0.0, particle["life"] - dt * 1.2)
            if new_life > 0:
                new_alpha = max(0, min(255, new_life * 255))
                new_paddle_particles.append(
                    {
                        **particle,
                        "life": new_life,
                        "alpha": new_alpha,
                    }
                )
        self.paddle_particles = new_paddle_particles

    def add_ball_particle(self, x, y, color):
        """
        Add a ball particle.
        Args:
            x (int): The x coordinate of the particle
            y (int): The y coordinate of the particle
            color (tuple): The color of the particle
        Returns:
            None
        """
        self.ball_particle_timer += 1.0
        if (
            self.ball_particle_timer >= self.ball_particle_interval
            and len(self.ball_particles) < self.max_ball_particles
        ):
            if isinstance(color, (tuple, list)) and len(color) >= 3:
                self.ball_particles.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "color": color[:3],
                        "life": 1.2,
                        "alpha": 255,
                    }
                )
                self.ball_particle_timer = 0.0

    def add_paddle_particle(self, x, y, color):
        """
        Add a paddle particle.
        Args:
            x (int): The x coordinate of the particle
            y (int): The y coordinate of the particle
            color (tuple): The color of the particle
        Returns:
            None
        """
        self.paddle_particle_timer += 1.0
        if (
            self.paddle_particle_timer >= self.paddle_particle_interval
            and len(self.paddle_particles) < self.max_paddle_particles
        ):
            if isinstance(color, (tuple, list)) and len(color) >= 3:
                self.paddle_particles.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "color": color[:3],
                        "life": 1.0,
                        "alpha": 200,
                    }
                )
                self.paddle_particle_timer = 0.0

    def draw_glow_rect(self, surface, color, rect, glow_size=10):
        """
        Draw a glow rectangle.
        Args:
            surface (pygame.Surface): The surface to draw on
            color (tuple): The color of the rectangle
            rect (pygame.Rect): The rectangle to draw
            glow_size (int): The size of the glow
        Returns:
            None
        """
        glow_color = tuple(c // 3 for c in color)
        max_glow_layers = min(glow_size, 6)
        glow_surf_size = (
            rect.width + max_glow_layers * 2,
            rect.height + max_glow_layers * 2,
        )
        glow_surf = pygame.Surface(glow_surf_size, pygame.SRCALPHA)
        for i in range(max_glow_layers, 0, -2):
            alpha = int(40 * (1 - i / max_glow_layers) * (0.5 + self.glow_pulse * 0.5))
            glow_rect = pygame.Rect(
                max_glow_layers - i,
                max_glow_layers - i,
                rect.width + i * 2,
                rect.height + i * 2,
            )
            pygame.draw.rect(glow_surf, (*glow_color, alpha), glow_rect)
        surface.blit(glow_surf, (rect.x - max_glow_layers, rect.y - max_glow_layers))
        pygame.draw.rect(surface, color, rect)
        if rect.width > 4 and rect.height > 4:
            highlight_color = tuple(min(255, c + 50) for c in color)
            inner_rect = pygame.Rect(
                rect.x + 1, rect.y + 1, rect.width - 2, rect.height - 2
            )
            pygame.draw.rect(surface, highlight_color, inner_rect, 1)

    def draw_glow_circle(self, surface, color, center, radius, glow_size=15):
        """
        Draw a glow circle.
        Args:
            surface (pygame.Surface): The surface to draw on
            color (tuple): The color of the circle
            center (tuple): The center of the circle
            radius (int): The radius of the circle
            glow_size (int): The size of the glow
        Returns:
            None
        """
        actual_glow_size = min(glow_size, 8)
        glow_color = tuple(c // 3 for c in color)
        surf_size = (
            radius * 2 + actual_glow_size * 2,
            radius * 2 + actual_glow_size * 2,
        )
        glow_surf = pygame.Surface(surf_size, pygame.SRCALPHA)
        for i in range(actual_glow_size, 0, -2):
            alpha = int(30 * (1 - i / actual_glow_size) * (0.5 + self.glow_pulse * 0.5))
            circle_pos = (radius + actual_glow_size, radius + actual_glow_size)
            pygame.draw.circle(glow_surf, (*glow_color, alpha), circle_pos, radius + i)
        blit_pos = (
            center[0] - radius - actual_glow_size,
            center[1] - radius - actual_glow_size,
        )
        surface.blit(glow_surf, blit_pos)
        pygame.draw.circle(surface, color, center, radius)
        if radius > 3:
            highlight_color = tuple(min(255, c + 80) for c in color)
            pygame.draw.circle(surface, highlight_color, center, max(1, radius - 2), 1)

    def draw_glow_line(self, surface, color, start_pos, end_pos, width=3, glow_size=8):
        """
        Draw a glow line.
        Args:
            surface (pygame.Surface): The surface to draw on
            color (tuple): The color of the line
            start_pos (tuple): The starting position of the line
            end_pos (tuple): The ending position of the line
            width (int): The width of the line
            glow_size (int): The size of the glow
        Returns:
            None
        """
        actual_glow_size = min(glow_size, 5)
        glow_color = tuple(c // 4 for c in color)
        line_length = (
            (end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2
        ) ** 0.5
        surf_width = int(line_length + actual_glow_size * 4)
        surf_height = width + actual_glow_size * 4
        glow_surf = pygame.Surface((surf_width, surf_height), pygame.SRCALPHA)
        for i in range(actual_glow_size, 0, -2):
            alpha = int(15 * (1 - i / actual_glow_size) * (0.5 + self.glow_pulse * 0.5))
            line_color = (*glow_color, alpha)
            line_width = width + i * 2
            rel_start = (actual_glow_size * 2, surf_height // 2)
            rel_end = (surf_width - actual_glow_size * 2, surf_height // 2)
            pygame.draw.line(glow_surf, line_color, rel_start, rel_end, line_width)
        import math

        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        rotated_surf = pygame.transform.rotate(glow_surf, -math.degrees(angle))
        rect = rotated_surf.get_rect()
        rect.center = (
            (start_pos[0] + end_pos[0]) // 2,
            (start_pos[1] + end_pos[1]) // 2,
        )
        surface.blit(rotated_surf, rect)
        pygame.draw.line(surface, color, start_pos, end_pos, width)

    def draw_gradient_background(self, surface, width, height):
        """
        Draw a gradient background.
        Args:
            surface (pygame.Surface): The surface to draw on
            width (int): The width of the background
            height (int): The height of the background
        Returns:
            None
        """
        base_color = self.DARK_BG
        surface.fill(base_color)
        wave_offset = math.sin(self.time_offset) * 20
        for i in range(5):
            alpha = int(20 + math.sin(self.time_offset + i) * 10)
            color = (
                min(255, base_color[0] + 10 + int(wave_offset)),
                min(255, base_color[1] + 15 + int(wave_offset)),
                min(255, base_color[2] + 25 + int(wave_offset)),
            )
            rect_surf = pygame.Surface((width // 3, height // 3), pygame.SRCALPHA)
            rect_surf.fill((*color, alpha))
            x = (width // 4) + int(math.sin(self.time_offset + i * 2) * 50)
            y = (height // 4) + int(math.cos(self.time_offset + i * 1.5) * 30)
            surface.blit(rect_surf, (x, y))

    def draw_particles(self, surface):
        """
        Draw the particles.
        Args:
            surface (pygame.Surface): The surface to draw on
        Returns:
            None
        """
        if not surface:
            return
        if not hasattr(self, "_ball_particle_surf"):
            self._ball_particle_surf = pygame.Surface((6, 6), pygame.SRCALPHA)
        if not hasattr(self, "_paddle_particle_surf"):
            self._paddle_particle_surf = pygame.Surface((4, 8), pygame.SRCALPHA)
        for particle in self.ball_particles:
            try:
                alpha = max(0, min(255, int(particle["alpha"])))
                if (
                    alpha > 0
                    and "x" in particle
                    and "y" in particle
                    and "color" in particle
                ):
                    self._ball_particle_surf.fill((0, 0, 0, 0))
                    color = particle["color"]
                    if len(color) >= 3:
                        color_with_alpha = (*color[:3], alpha)
                        pygame.draw.circle(
                            self._ball_particle_surf, color_with_alpha, (3, 3), 3
                        )
                        surface.blit(
                            self._ball_particle_surf,
                            (int(particle["x"]) - 3, int(particle["y"]) - 3),
                        )
            except (ValueError, TypeError, KeyError):
                continue
        for particle in self.paddle_particles:
            try:
                alpha = max(0, min(255, int(particle["alpha"])))
                if (
                    alpha > 0
                    and "x" in particle
                    and "y" in particle
                    and "color" in particle
                ):
                    self._paddle_particle_surf.fill((0, 0, 0, 0))
                    color = particle["color"]
                    if len(color) >= 3:
                        color_with_alpha = (*color[:3], alpha)
                        pygame.draw.rect(
                            self._paddle_particle_surf, color_with_alpha, (0, 0, 4, 8)
                        )
                        surface.blit(
                            self._paddle_particle_surf,
                            (int(particle["x"]) - 2, int(particle["y"]) - 4),
                        )
            except (ValueError, TypeError, KeyError):
                continue

    def draw_neon_text(self, surface, font, text, color, pos, glow_size=5):
        """
        Draw neon text.
        Args:
            surface (pygame.Surface): The surface to draw on
            font (pygame.Font): The font to use
            text (str): The text to draw
            color (tuple): The color of the text
            pos (tuple): The position of the text
            glow_size (int): The size of the glow
        Returns:
            pygame.Rect: The rectangle of the text
        """
        glow_color = tuple(c // 4 for c in color)
        for i in range(glow_size, 0, -1):
            alpha = int(40 * (1 - i / glow_size))
            glow_surf = font.render(text, True, glow_color)
            glow_surf.set_alpha(alpha)
            for dx in range(-i, i + 1):
                for dy in range(-i, i + 1):
                    if dx * dx + dy * dy <= i * i:
                        surface.blit(glow_surf, (pos[0] + dx, pos[1] + dy))
        text_surf = font.render(text, True, color)
        surface.blit(text_surf, pos)
        return text_surf.get_rect(topleft=pos)


class PauseMenu:
    """
    Pause menu for the Pong game.
    """

    def __init__(self, screen, width, height):
        """
        Initialize the PauseMenu class.
        Args:
            screen (pygame.Surface): The screen to draw on
            width (int): The width of the menu
            height (int): The height of the menu
        """
        self.screen = screen
        self.width = width
        self.height = height
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.BLUE = (0, 100, 200)
        self.BLACK = (0, 0, 0)
        self.menu_items = [
            "Continue",
            "Save AI",
            "Reset AI",
            "Settings",
            "Exit to Menu",
        ]
        self.selected_index = 0

    def handle_events(self, event):
        """
        Handle events for the pause menu.
        Args:
            event (pygame.Event): The event to handle
        Returns:
            str: The selected item
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_index = (self.selected_index - 1) % len(self.menu_items)
            elif event.key == pygame.K_DOWN:
                self.selected_index = (self.selected_index + 1) % len(self.menu_items)
            elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                return self.menu_items[self.selected_index]
            elif event.key == pygame.K_ESCAPE:
                return "Continue"
        return None

    def draw(self):
        """
        Draw the pause menu.
        Returns:
            None
        """
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(128)
        overlay.fill(self.BLACK)
        self.screen.blit(overlay, (0, 0))
        game = getattr(self, "game", None)
        use_neon = getattr(game, "use_neon_theme", False) and hasattr(
            game, "neon_theme"
        )
        if use_neon and game.neon_theme:
            title_color = game.neon_theme.NEON_ORANGE
            title_text = "PAUSE"
            title_width = self.font_large.size(title_text)[0]
            title_x = self.width // 2 - title_width // 2
            game.neon_theme.draw_neon_text(
                self.screen,
                self.font_large,
                title_text,
                title_color,
                (title_x, self.height // 2 - 150),
            )
        else:
            title = self.font_large.render("PAUSE", True, self.WHITE)
            title_rect = title.get_rect(
                center=(self.width // 2, self.height // 2 - 150)
            )
            self.screen.blit(title, title_rect)
        start_y = self.height // 2 - 80
        for i, item in enumerate(self.menu_items):
            is_selected = i == self.selected_index
            if use_neon and game.neon_theme:
                color = (
                    game.neon_theme.NEON_PINK
                    if is_selected
                    else game.neon_theme.NEON_CYAN
                )
                item_text = item
                text_width = self.font_medium.size(item_text)[0]
                text_x = self.width // 2 - text_width // 2
                text_y = start_y + i * 40 - self.font_medium.get_height() // 2
                if is_selected:
                    item_rect = pygame.Rect(
                        text_x - 20, text_y - 10, text_width + 40, 30
                    )
                    game.neon_theme.draw_glow_rect(
                        self.screen, game.neon_theme.NEON_PURPLE, item_rect, glow_size=6
                    )
                game.neon_theme.draw_neon_text(
                    self.screen, self.font_medium, item_text, color, (text_x, text_y)
                )
            else:
                color = self.BLUE if is_selected else self.WHITE
                text = self.font_medium.render(item, True, color)
                text_rect = text.get_rect(center=(self.width // 2, start_y + i * 40))
                self.screen.blit(text, text_rect)


class MetricsDisplay:
    def __init__(self, game):
        """
        Initialize the MetricsDisplay class.
        Args:
            game (Game): The game instance
        """
        self.game = game
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 32)
        self.metrics_update_interval = 0.1
        self.last_update = 0
        self.previous_metrics = {"player": None, "opponent": None}
        self.transition_speed = 0.3
        self.left_metrics_x = 10
        self.right_metrics_x = self.game.WIDTH - 210
        self.metrics_y = 10
        self.metrics_width = 200
        self.metrics_height = 150

    def draw(self, surface):
        """
        Draw the metrics display.
        Args:
            surface (pygame.Surface): The surface to draw on
        Returns:
            None
        """
        current_time = time.time()
        player_metrics = None
        opponent_metrics = None
        if self.game.player.ai_controller:
            player_metrics = self.game.player.ai_controller.get_performance_metrics()
        if self.game.opponent.ai_controller:
            opponent_metrics = (
                self.game.opponent.ai_controller.get_performance_metrics()
            )
        if current_time - self.last_update >= self.metrics_update_interval:
            self.last_update = current_time
            if player_metrics:
                if self.previous_metrics["player"] is None:
                    self.previous_metrics["player"] = player_metrics
                else:
                    self.previous_metrics["player"] = self._smooth_transition(
                        self.previous_metrics["player"], player_metrics
                    )
            if opponent_metrics:
                if self.previous_metrics["opponent"] is None:
                    self.previous_metrics["opponent"] = opponent_metrics
                else:
                    self.previous_metrics["opponent"] = self._smooth_transition(
                        self.previous_metrics["opponent"], opponent_metrics
                    )
        if self.previous_metrics["player"]:
            self._draw_metrics_section(
                surface,
                "Player",
                self.previous_metrics["player"],
                self.left_metrics_x,
                self.metrics_y,
            )
        if self.previous_metrics["opponent"]:
            self._draw_metrics_section(
                surface,
                "Opponent",
                self.previous_metrics["opponent"],
                self.right_metrics_x,
                self.metrics_y,
            )

    def _smooth_transition(self, old_metrics, new_metrics):
        """
        Smoothly transition between old and new metrics.
        Args:
            old_metrics (dict): The old metrics
            new_metrics (dict): The new metrics
        Returns:
            dict: The transitioned metrics
        """
        if old_metrics is None:
            return new_metrics
        transitioned = {}
        for key in new_metrics:
            if isinstance(new_metrics[key], (int, float)):
                old_val = old_metrics.get(key, new_metrics[key])
                new_val = new_metrics[key]
                transitioned[key] = (
                    old_val + (new_val - old_val) * self.transition_speed
                )
            else:
                transitioned[key] = new_metrics[key]
        return transitioned

    def _draw_metrics_section(self, surface, title, metrics, x, y):
        """
        Draw a metrics section.
        Args:
            surface (pygame.Surface): The surface to draw on
            title (str): The title of the section
            metrics (dict): The metrics to draw
            x (int): The x coordinate of the section
            y (int): The y coordinate of the section
        Returns:
            None
        """
        title_text = self.font.render(f"{title} Metrics:", True, (200, 200, 200))
        surface.blit(title_text, (x, y))
        y += 30
        metrics_to_show = [
            (
                "Accuracy",
                metrics["round_accuracy"],
                self._get_accuracy_color(metrics["round_accuracy"]),
            ),
            (
                "Long-term",
                metrics["long_term_accuracy"],
                self._get_accuracy_color(metrics["long_term_accuracy"]),
            ),
            (
                "Recent",
                metrics["recent_performance"],
                self._get_accuracy_color(metrics["recent_performance"]),
            ),
            ("Streak", metrics["best_consecutive_hits"], (255, 255, 255)),
            ("Diff", f"{metrics['current_difficulty']:.1f}x", (255, 255, 255)),
            ("Episodes", metrics["total_episodes"], (255, 255, 255)),
        ]
        for label, value, color in metrics_to_show:
            text = self.font.render(
                (
                    f"{label}: {value:.2f}"
                    if isinstance(value, float)
                    else f"{label}: {value}"
                ),
                True,
                color,
            )
            surface.blit(text, (x, y))
            y += 20

    def _get_accuracy_color(self, accuracy):
        """
        Get the accuracy color.
        Args:
            accuracy (float): The accuracy
        Returns:
            tuple: The color
        """
        if accuracy >= 0.7:
            return (0, 255, 0)
        elif accuracy >= 0.4:
            return (255, 255, 0)
        else:
            return (255, 0, 0)


class Paddle:
    def __init__(self, x, y, width, height, speed, game, is_ai=False):
        """
        Initialize the Paddle class.
        Args:
            x (int): The x coordinate of the paddle
            y (int): The y coordinate of the paddle
            width (int): The width of the paddle
            height (int): The height of the paddle
            speed (int): The speed of the paddle
            game (Game): The game instance
            is_ai (bool): Whether the paddle is AI-controlled
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.speed = speed
        self.game = game
        self.is_ai = is_ai
        self.ai_controller = None
        self.moving = False
        self.last_y = y

    def move(self, direction):
        """
        Move the paddle.
        Args:
            direction (int): The direction to move
        Returns:
            None
        """
        if direction == 0:
            self.moving = False
            return
        self.last_y = self.rect.y
        self.rect.y += direction * self.speed
        self.rect.y = max(0, min(self.rect.y, self.game.HEIGHT - self.rect.height))
        self.moving = self.rect.y != self.last_y

    def draw(self, screen, color):
        """
        Draw the paddle.
        Args:
            screen (pygame.Surface): The screen to draw on
            color (tuple): The color of the paddle
        Returns:
            None
        """
        pygame.draw.rect(screen, color, self.rect)

    def update(self):
        """
        Update the paddle.
        Returns:
            None
        """
        if self.is_ai and self.ai_controller:
            self.ai_controller.update()
        elif not self.is_ai:
            keys = pygame.key.get_pressed()
            if hasattr(self.game, "two_player_mode") and self.game.two_player_mode:
                if self == self.game.opponent:
                    self.move(keys[pygame.K_s] - keys[pygame.K_w])
                else:
                    self.move(keys[pygame.K_DOWN] - keys[pygame.K_UP])
            else:
                self.move(keys[pygame.K_DOWN] - keys[pygame.K_UP])


class Ball:
    def __init__(self, x, y, size, speed_x, speed_y):
        """
        Initialize the Ball class.
        Args:
            x (int): The x coordinate of the ball
            y (int): The y coordinate of the ball
            size (int): The size of the ball
            speed_x (int): The x speed of the ball
            speed_y (int): The y speed of the ball
        Returns:
            None
        """
        self.rect = pygame.Rect(x, y, size, size)
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.max_speed = max(abs(speed_x), abs(speed_y))
        self.last_collision_time = 0
        self.collision_cooldown = 0.001

    def move(self):
        """
        Move the ball.
        Returns:
            None
        """
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

    def bounce(self, top, bottom, left, right):
        """
        Bounce the ball.
        Args:
            top (int): The top of the screen
            bottom (int): The bottom of the screen
            left (int): The left of the screen
            right (int): The right of the screen
        Returns:
            None
        """
        current_time = time.time()
        if current_time - self.last_collision_time < self.collision_cooldown:
            return
        if self.rect.top <= top:
            self.rect.top = top
            self.speed_y = abs(self.speed_y)
            self.last_collision_time = current_time
        elif self.rect.bottom >= bottom:
            self.rect.bottom = bottom
            self.speed_y = -abs(self.speed_y)
            self.last_collision_time = current_time
        if self.rect.left <= left:
            self.rect.left = left
            self.speed_x = abs(self.speed_x)
            self.last_collision_time = current_time
        elif self.rect.right >= right:
            self.rect.right = right
            self.speed_x = -abs(self.speed_x)
            self.last_collision_time = current_time

    def handle_paddle_collision(self, paddle):
        """
        Handle paddle collision.
        Args:
            paddle (Paddle): The paddle to collide with
        Returns:
            bool: True if a collision occurred, False otherwise
        """
        current_time = time.time()
        if current_time - self.last_collision_time < self.collision_cooldown:
            return False
        relative_intersect_y = (paddle.rect.centery - self.rect.centery) / (
            paddle.rect.height / 2
        )
        bounce_angle = relative_intersect_y * 45
        bounce_angle_rad = math.radians(bounce_angle)
        speed = math.sqrt(self.speed_x**2 + self.speed_y**2)
        is_left_paddle = paddle.rect.centerx < 400
        if is_left_paddle:
            self.speed_x = abs(speed * math.cos(bounce_angle_rad))
        else:
            self.speed_x = -abs(speed * math.cos(bounce_angle_rad))
        self.speed_y = -speed * math.sin(bounce_angle_rad)
        min_vertical_speed = speed * 0.2
        if abs(self.speed_y) < min_vertical_speed:
            self.speed_y = min_vertical_speed * (1 if self.speed_y >= 0 else -1)
            horizontal_component = math.sqrt(speed**2 - self.speed_y**2)
            if is_left_paddle:
                self.speed_x = horizontal_component
            else:
                self.speed_x = -horizontal_component
        if is_left_paddle:
            self.rect.left = paddle.rect.right + 2
        else:
            self.rect.right = paddle.rect.left - 2
        self.speed_x += 0.1
        self.speed_y += 0.1
        self.last_collision_time = current_time
        return True

    def reset(self, x, y):
        """
        Reset the ball.
        Args:
            x (int): The x coordinate of the ball
            y (int): The y coordinate of the ball
        Returns:
            None
        """
        self.rect.center = (x, y)
        angle = random.uniform(-45, 45)
        angle_rad = math.radians(angle)
        speed = math.sqrt(self.speed_x**2 + self.speed_y**2)
        self.speed_x = speed * math.cos(angle_rad)
        self.speed_y = speed * math.sin(angle_rad)
        self.speed_x *= random.choice([-1, 1])
        min_vertical_speed = speed * 0.2
        if abs(self.speed_y) < min_vertical_speed:
            self.speed_y = min_vertical_speed * (1 if self.speed_y >= 0 else -1)
            horizontal_component = math.sqrt(speed**2 - self.speed_y**2)
            self.speed_x = horizontal_component * (1 if self.speed_x >= 0 else -1)

    def draw(self, screen, color):
        """
        Draw the ball.
        Args:
            screen (pygame.Surface): The screen to draw on
            color (tuple): The color of the ball
        Returns:
            None
        """
        pygame.draw.ellipse(screen, color, self.rect)


class Score:
    def __init__(self, font):
        """
        Initialize the Score class.
        Args:
            font (pygame.Font): The font to use
        """
        self.font = font
        self.player_score = 0
        self.opponent_score = 0

    def update(self, player_scored, opponent_scored):
        """
        Update the score.
        Args:
            player_scored (bool): Whether the player scored
            opponent_scored (bool): Whether the opponent scored
        Returns:
            None
        """
        if player_scored:
            self.player_score += 1
        if opponent_scored:
            self.opponent_score += 1

    def draw(self, screen, color, width):
        """
        Draw the score.
        Args:
            screen (pygame.Surface): The screen to draw on
            color (tuple): The color of the score
            width (int): The width of the screen
        Returns:
            None
        """
        player_text = self.font.render(str(self.player_score), False, color)
        opponent_text = self.font.render(str(self.opponent_score), False, color)
        screen.blit(player_text, (width // 4, 20))
        screen.blit(opponent_text, (3 * width // 4, 20))


class RenderThread(threading.Thread):
    def __init__(self, game):
        """
        Initialize the RenderThread class.
        Args:
            game (Game): The game instance
        """
        super().__init__()
        self.game = game
        self.running = True
        self.render_queue = queue.Queue()
        self.daemon = True

    def run(self):
        """
        Run the render thread.
        Returns:
            None
        """
        pygame.init()
        while self.running:
            try:
                if not self.render_queue.empty():
                    while not self.render_queue.empty():
                        self.render_queue.get_nowait()
                    self.game.screen.fill((0, 0, 0))
                    self.game.draw()
                    pygame.display.flip()
                time.sleep(1 / 120)
            except Exception as e:
                print(f"Rendering thread error: {e}")

    def stop(self):
        """
        Stop the render thread.
        Returns:
            None
        """
        self.running = False
        try:
            while not self.render_queue.empty():
                self.render_queue.get_nowait()
        except Exception:
            pass


class PowerUpManager:
    def __init__(self, game):
        """
        Initialize the PowerUpManager class.
        Args:
            game (Game): The game instance
        """
        self.game = game
        self.player1_combo_count = 0
        self.player1_combo_multiplier = 1.0
        self.player2_combo_count = 0
        self.player2_combo_multiplier = 1.0
        self.max_combo_multiplier = 3.0
        self.combo_count = 0
        self.combo_multiplier = 1.0
        self.player_powerups = {
            "size_boost": {"active": False, "timer": 0, "duration": 600},
            "speed_boost": {"active": False, "timer": 0, "duration": 300},
        }
        self.opponent_powerups = {
            "size_boost": {"active": False, "timer": 0, "duration": 600},
            "speed_boost": {"active": False, "timer": 0, "duration": 300},
        }
        self.ball_powerups = {
            "slowdown": {"active": False, "timer": 0, "duration": 300},
            "speedup": {"active": False, "timer": 0, "duration": 180},
            "multiball": {"active": False, "timer": 0, "duration": 600},
            "chaotic": {"active": False, "timer": 0, "duration": 480},
        }
        self.chaos_change_interval = 120
        self.chaos_last_change = 0
        self.chaos_colors = [
            (255, 100, 100),
            (100, 255, 100),
            (100, 100, 255),
            (255, 255, 100),
            (255, 100, 255),
            (100, 255, 255),
        ]
        self.chaos_color_index = 0
        self.original_paddle_height = game.PADDLE_HEIGHT
        self.original_paddle_speed = game.PADDLE_SPEED
        self.original_ball_speed = (game.BALL_SPEED_X, game.BALL_SPEED_Y)
        self.powerup_chance = 0.15
        self.total_hits = 0
        self.successful_combos = 0

    def update_combo(
        self,
        player_hit=False,
        player2_hit=False,
        reset_player1=False,
        reset_player2=False,
        reset=False,
    ):
        """
        Update the combo.
        Args:
            player_hit (bool): Whether the player hit
            player2_hit (bool): Whether the player 2 hit
            reset_player1 (bool): Whether to reset the player 1 combo
            reset_player2 (bool): Whether to reset the player 2 combo
            reset (bool): Whether to reset the combo
        Returns:
            None
        """
        if reset:
            reset_player1 = True
        if reset_player1:
            if self.player1_combo_count > 0:
                print(f"Player 1 combo broken! Was: x{self.player1_combo_count}")
            self.player1_combo_count = 0
            self.player1_combo_multiplier = 1.0
        elif player_hit:
            self.player1_combo_count += 1
            self.player1_combo_multiplier = min(
                1.0 + (self.player1_combo_count * 0.2), self.max_combo_multiplier
            )
            if self.player1_combo_count == 3:
                print(
                    f"Player 1 combo x{self.player1_combo_count}! "
                    f"Multiplier: x{self.player1_combo_multiplier:.1f}"
                )
            elif self.player1_combo_count > 3 and self.player1_combo_count % 5 == 0:
                print(
                    f"Player 1 SUPER-COMBO x{self.player1_combo_count}! "
                    f"Multiplier: x{self.player1_combo_multiplier:.1f}"
                )
        if reset_player2:
            if self.player2_combo_count > 0:
                print(f"Player 2 combo broken! Was: x{self.player2_combo_count}")
            self.player2_combo_count = 0
            self.player2_combo_multiplier = 1.0
        elif player2_hit:
            self.player2_combo_count += 1
            self.player2_combo_multiplier = min(
                1.0 + (self.player2_combo_count * 0.2), self.max_combo_multiplier
            )
            if self.player2_combo_count == 3:
                print(
                    f"Player 2 combo x{self.player2_combo_count}! "
                    f"Multiplier: x{self.player2_combo_multiplier:.1f}"
                )
            elif self.player2_combo_count > 3 and self.player2_combo_count % 5 == 0:
                print(
                    f"Player 2 SUPER-COMBO x{self.player2_combo_count}! "
                    f"Multiplier: x{self.player2_combo_multiplier:.1f}"
                )
        self.combo_count = self.player1_combo_count
        self.combo_multiplier = self.player1_combo_multiplier
        if self.player1_combo_count >= 3:
            self.successful_combos += 1

    def activate_powerup(self, target, powerup_type):
        """
        Activate a powerup.
        Args:
            target (str): The target of the powerup
            powerup_type (str): The type of powerup
        Returns:
            bool: True if the powerup was activated, False otherwise
        """
        if target == "player":
            powerups = self.player_powerups
            paddle = self.game.player
        elif target == "opponent":
            powerups = self.opponent_powerups
            paddle = self.game.opponent
        elif target == "ball":
            powerups = self.ball_powerups
            paddle = None
        else:
            return False
        if powerup_type in powerups and not powerups[powerup_type]["active"]:
            powerups[powerup_type]["active"] = True
            powerups[powerup_type]["timer"] = powerups[powerup_type]["duration"]
            if target != "ball" and paddle:
                if powerup_type == "size_boost":
                    paddle.rect.height = int(self.original_paddle_height * 1.5)
                    if paddle.rect.bottom > self.game.HEIGHT:
                        paddle.rect.bottom = self.game.HEIGHT
                elif powerup_type == "speed_boost":
                    paddle.speed = int(self.original_paddle_speed * 1.5)
            elif target == "ball":
                if powerup_type == "slowdown":
                    for ball in self.game.balls:
                        ball.speed_x *= 0.6
                        ball.speed_y *= 0.6
                elif powerup_type == "speedup":
                    for ball in self.game.balls:
                        ball.speed_x *= 1.4
                        ball.speed_y *= 1.4
                elif powerup_type == "multiball":
                    self._activate_multiball()
                elif powerup_type == "chaotic":
                    self._activate_chaotic_ball()
            print(f"Activated {powerup_type} power-up for {target}")
            return True
        return False

    def update(self):
        """
        Update the powerups.
        Returns:
            None
        """
        for powerup_type, data in self.player_powerups.items():
            if data["active"]:
                data["timer"] -= 1
                if data["timer"] <= 0:
                    self._deactivate_powerup("player", powerup_type)
        for powerup_type, data in self.opponent_powerups.items():
            if data["active"]:
                data["timer"] -= 1
                if data["timer"] <= 0:
                    self._deactivate_powerup("opponent", powerup_type)
        for powerup_type, data in self.ball_powerups.items():
            if data["active"]:
                data["timer"] -= 1
                if data["timer"] <= 0:
                    self._deactivate_powerup("ball", powerup_type)

    def _deactivate_powerup(self, target, powerup_type):
        """
        Deactivate a powerup.
        Args:
            target (str): The target of the powerup
            powerup_type (str): The type of powerup
        Returns:
            None
        """
        if target == "player":
            powerups = self.player_powerups
            paddle = self.game.player
        elif target == "opponent":
            powerups = self.opponent_powerups
            paddle = self.game.opponent
        elif target == "ball":
            powerups = self.ball_powerups
            paddle = None
        powerups[powerup_type]["active"] = False
        if target != "ball" and paddle:
            if powerup_type == "size_boost":
                paddle.rect.height = self.original_paddle_height
            elif powerup_type == "speed_boost":
                paddle.speed = self.original_paddle_speed
        elif target == "ball":
            if powerup_type in ["slowdown", "speedup"]:
                current_speed = (
                    self.game.ball.speed_x**2 + self.game.ball.speed_y**2
                ) ** 0.5
                original_speed = (
                    self.original_ball_speed[0] ** 2 + self.original_ball_speed[1] ** 2
                ) ** 0.5
                if current_speed > 0:
                    speed_ratio = original_speed / current_speed
                    self.game.ball.speed_x *= speed_ratio
                    self.game.ball.speed_y *= speed_ratio
        print(f"Deactivated {powerup_type} power-up for {target}")

    def _activate_multiball(self):
        """
        Activate a multiball.
        Returns:
            bool: True if the multiball was activated, False otherwise
        """
        import math
        import random

        current_balls = len(self.game.balls)
        if current_balls >= self.game.max_balls:
            print("Maximum number of balls already reached")
            return False
        balls_to_add = min(2, self.game.max_balls - current_balls)
        for i in range(balls_to_add):
            new_ball = Ball(
                self.game.WIDTH // 2 - self.game.BALL_SIZE // 2,
                self.game.HEIGHT // 2 - self.game.BALL_SIZE // 2,
                self.game.BALL_SIZE,
                self.game.BALL_SPEED_X,
                self.game.BALL_SPEED_Y,
            )
            angle = random.uniform(-45, 45)
            angle_rad = math.radians(angle)
            speed = math.sqrt(new_ball.speed_x**2 + new_ball.speed_y**2)
            new_ball.speed_x = speed * math.cos(angle_rad) * random.choice([-1, 1])
            new_ball.speed_y = speed * math.sin(angle_rad)
            offset = (i + 1) * 30
            new_ball.rect.x += random.randint(-offset, offset)
            new_ball.rect.y += random.randint(-offset, offset)
            self.game.balls.append(new_ball)
        print(
            f"Multiball! Added {balls_to_add} balls. " f"Total: {len(self.game.balls)}"
        )
        return True

    def _activate_chaotic_ball(self):
        """
        Activate a chaotic ball.
        Returns:
            bool: True if the chaotic ball was activated, False otherwise
        """
        print("Chaotic ball activated!")
        self.chaos_last_change = 0
        self.chaos_color_index = 0
        for ball in self.game.balls:
            self._apply_chaos_to_ball(ball)
        return True

    def _apply_chaos_to_ball(self, ball):
        """
        Apply chaos to a ball.
        Args:
            ball (Ball): The ball to apply chaos to
        Returns:
            None
        """
        import math
        import random

        current_speed = math.sqrt(ball.speed_x**2 + ball.speed_y**2)
        angle = random.uniform(-60, 60)
        angle_rad = math.radians(angle)
        speed_multiplier = random.uniform(0.7, 1.5)
        new_speed = current_speed * speed_multiplier
        direction = random.choice([-1, 1])
        ball.speed_x = direction * new_speed * math.cos(angle_rad)
        ball.speed_y = new_speed * math.sin(angle_rad)
        print(f"Chaos! New speed: ({ball.speed_x:.1f}, {ball.speed_y:.1f})")

    def _update_chaotic_ball(self):
        """
        Update the chaotic ball.
        Returns:
            None
        """
        self.chaos_last_change += 1
        if self.chaos_last_change % 30 == 0:
            self.chaos_color_index = (self.chaos_color_index + 1) % len(
                self.chaos_colors
            )
        if self.chaos_last_change >= self.chaos_change_interval:
            for ball in self.game.balls:
                self._apply_chaos_to_ball(ball)
            self.chaos_last_change = 0

    def get_chaos_color(self):
        """
        Get the chaos color.
        Returns:
            tuple: The chaos color
        """
        if self.ball_powerups["chaotic"]["active"]:
            return self.chaos_colors[self.chaos_color_index]
        return None

    def try_spawn_powerup(self, event_type="hit"):
        """
        Try to spawn a powerup.
        Args:
            event_type (str): The type of event
        Returns:
            bool: True if the powerup was spawned, False otherwise
        """
        import random

        if random.random() < self.powerup_chance:
            powerup_options = [
                ("player", "size_boost"),
                ("player", "speed_boost"),
                ("ball", "slowdown"),
                ("opponent", "speed_boost"),
            ]
            if event_type == "combo" and self.combo_count >= 3:
                powerup_options = [
                    ("player", "size_boost"),
                    ("player", "speed_boost"),
                    ("ball", "slowdown"),
                ] * 3 + powerup_options
            target, powerup_type = random.choice(powerup_options)
            return self.activate_powerup(target, powerup_type)
        return False

    def get_ai_state_data(self):
        """
        Get the AI state data.
        Returns:
            dict: The AI state data
        """
        return {
            "combo_multiplier": self.combo_multiplier,
            "combo_count": self.combo_count,
            "player_size_boost": self.player_powerups["size_boost"]["active"],
            "player_speed_boost": self.player_powerups["speed_boost"]["active"],
            "opponent_size_boost": self.opponent_powerups["size_boost"]["active"],
            "opponent_speed_boost": self.opponent_powerups["speed_boost"]["active"],
            "ball_slowdown": self.ball_powerups["slowdown"]["active"],
            "ball_speedup": self.ball_powerups["speedup"]["active"],
            "paddle_size_ratio": (
                self.game.player.rect.height / self.original_paddle_height
            ),
            "ball_speed_ratio": (
                (self.game.ball.speed_x**2 + self.game.ball.speed_y**2) ** 0.5
                / (self.original_ball_speed[0] ** 2 + self.original_ball_speed[1] ** 2)
                ** 0.5
            ),
        }

    def draw_powerup_indicators(self, screen):
        """
        Draw the powerup indicators.
        Args:
            screen (pygame.Surface): The screen to draw on
        Returns:
            None
        """
        font = pygame.font.Font(None, 24)
        y_offset = 120
        if self.player1_combo_count > 0:
            combo_text = f"P1 STREAK x{self.player1_combo_count} (x{self.player1_combo_multiplier:.1f})"
            if self.player1_combo_count >= 10:
                color = (100, 100, 255)
            elif self.player1_combo_count >= 5:
                color = (0, 150, 255)
            elif self.player1_combo_count >= 3:
                color = (100, 200, 255)
            else:
                color = (200, 200, 255)
            text_surface = font.render(combo_text, True, color)
            screen.blit(text_surface, (10, y_offset))
            y_offset += 25
        if self.player2_combo_count > 0:
            combo_text = f"P2 STREAK x{self.player2_combo_count} (x{self.player2_combo_multiplier:.1f})"
            if self.player2_combo_count >= 10:
                color = (255, 100, 100)
            elif self.player2_combo_count >= 5:
                color = (255, 150, 0)
            elif self.player2_combo_count >= 3:
                color = (255, 200, 100)
            else:
                color = (255, 200, 200)
            text_surface = font.render(combo_text, True, color)
            screen.blit(text_surface, (10, y_offset))
            y_offset += 25
        for powerup_type, data in self.player_powerups.items():
            if data["active"]:
                time_left = data["timer"] // 60
                text = f"Player {powerup_type}: {time_left}s"
                text_surface = font.render(text, True, (0, 255, 0))
                screen.blit(text_surface, (10, y_offset))
                y_offset += 25
        if len(self.game.balls) > 1:
            text = f"Balls: {len(self.game.balls)}"
            text_surface = font.render(text, True, (255, 100, 255))
            screen.blit(text_surface, (10, y_offset))
            y_offset += 25
        for powerup_type, data in self.ball_powerups.items():
            if data["active"]:
                time_left = data["timer"] // 60
                if powerup_type == "multiball":
                    text = f"Multiball: {time_left}s"
                    color = (255, 100, 255)
                elif powerup_type == "slowdown":
                    text = f"Slowdown: {time_left}s"
                    color = (100, 150, 255)
                elif powerup_type == "speedup":
                    text = f"Speedup: {time_left}s"
                    color = (255, 150, 100)
                elif powerup_type == "chaotic":
                    text = f"Chaos: {time_left}s"
                    color = (255, 255, 100)
                else:
                    text = f"Ball {powerup_type}: {time_left}s"
                    color = (255, 255, 0)
                text_surface = font.render(text, True, color)
                screen.blit(text_surface, (10, y_offset))
                y_offset += 25


class Game:
    def __init__(self, training_mode=False):
        """
        Initialize the Game class.
        Args:
            training_mode (bool): Whether to enable training mode
        """
        pygame.init()
        self.WIDTH, self.HEIGHT = 1200, 900
        self.neon_theme = NeonTheme()
        self.neon_theme.game = self
        self.use_neon_theme = True
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 30, 110
        self.BALL_SIZE = 20
        self.PADDLE_SPEED = 10
        self.BALL_SPEED_X, self.BALL_SPEED_Y = (
            5,
            5,
        )
        self.FPS = 120
        self.frame_time = 1.0 / self.FPS
        self.simulation_speed = 10.0 if training_mode else 1.0
        self.fast_training = training_mode
        self.two_player_mode = False
        self.training_mode = training_mode
        self.return_to_menu = False
        self.paused = False
        self.pause_menu_active = False
        self.show_metrics = False
        self._resources_initialized = False
        self.round_goals = 0
        self.rounds_completed = 0
        self.goals_per_round = 50
        try:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Pong - AI vs AI (Click to focus!)")
            self.state_lock = threading.Lock()
            try:
                import os

                if os.system == "darwin":
                    pass
            except Exception as e:
                print(f"Error bringing window to front: {e}")
                pass
            self.player = Paddle(
                50,
                self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2,
                self.PADDLE_WIDTH,
                self.PADDLE_HEIGHT,
                self.PADDLE_SPEED,
                self,
                is_ai=training_mode,
            )
            self.opponent = Paddle(
                self.WIDTH - 50 - self.PADDLE_WIDTH,
                self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2,
                self.PADDLE_WIDTH,
                self.PADDLE_HEIGHT,
                self.PADDLE_SPEED,
                self,
                is_ai=True,
            )
            initial_ball = Ball(
                self.WIDTH // 2 - self.BALL_SIZE // 2,
                self.HEIGHT // 2 - self.BALL_SIZE // 2,
                self.BALL_SIZE,
                self.BALL_SPEED_X,
                self.BALL_SPEED_Y,
            )
            initial_ball.reset(self.WIDTH // 2, self.HEIGHT // 2)
            self.balls = [initial_ball]
            self.max_balls = 3
            self.ball = self.balls[0]
            self.score = Score(pygame.font.Font(None, 36))
            self.last_frame_time = time.time()
            self.pause_menu = PauseMenu(self.screen, self.WIDTH, self.HEIGHT)
            self.pause_menu.game = self
            self.metrics_display = MetricsDisplay(self)
            if not self.two_player_mode:
                print("Initializing parallel AI controller...")
                self.opponent.ai_controller = ParallelAIController(
                    self.opponent,
                    self.ball,
                    num_agents=2,
                    input_size=12,
                    hidden_size=32,
                    output_size=1,
                    exchange_interval=2000,
                    mutation_rate=0.05,
                    batch_size=16,
                )
                print("Parallel AI controller initialized successfully")
                if self.opponent.ai_controller:
                    self.opponent.ai_controller.start_training()
            if self.training_mode and self.player.is_ai:
                print("Initializing player AI controller...")
                self.player.ai_controller = ParallelAIController(
                    self.player,
                    self.ball,
                    num_agents=2,
                    input_size=12,
                    hidden_size=32,
                    output_size=1,
                    exchange_interval=2000,
                    mutation_rate=0.05,
                    batch_size=16,
                )
                print("Player AI controller initialized successfully")
                if self.player.ai_controller:
                    self.player.ai_controller.start_training()
            self.render_thread = None
            self.powerup_manager = PowerUpManager(self)
            self.nn_visualizer_manager = NeuralNetworkVisualizerManager(
                max_visualizers=2
            )
            self.show_nn_visualization = False
            if self.opponent.is_ai:
                opponent_x = max(
                    self.WIDTH // 2 + 50,
                    self.opponent.rect.x + self.opponent.rect.width + 20,
                )
                opponent_y = 50
                self.nn_visualizer_manager.add_visualizer(
                    "opponent", position=(opponent_x, opponent_y)
                )
            if self.player.is_ai:
                player_x = max(
                    20,
                    min(
                        self.player.rect.x + self.player.rect.width + 20,
                        self.WIDTH // 2 - 280,
                    ),
                )
                player_y = 50
                self.nn_visualizer_manager.add_visualizer(
                    "player", position=(player_x, player_y)
                )
            self._resources_initialized = True
        except Exception as e:
            print(f"Error initializing game: {e}")
            self.cleanup()
            raise

    def cleanup(self):
        """
        Cleanup the game.
        Returns:
            None
        """
        try:
            if hasattr(self, "render_thread") and self.render_thread:
                self.render_thread.stop()
                self.render_thread.join(timeout=2.0)
                if self.render_thread.is_alive():
                    print("Warning: Render thread did not stop cleanly")
            if (
                hasattr(self, "opponent")
                and self.opponent
                and self.opponent.ai_controller
            ):
                try:
                    self.opponent.ai_controller.stop_training()
                    if hasattr(self.opponent.ai_controller, "_save_best_agent"):
                        self.opponent.ai_controller._save_best_agent()
                except Exception as e:
                    print(f"Error stopping opponent AI: {e}")
            if hasattr(self, "player") and self.player and self.player.ai_controller:
                try:
                    self.player.ai_controller.stop_training()
                    if hasattr(self.player.ai_controller, "_save_best_agent"):
                        self.player.ai_controller._save_best_agent()
                except Exception as e:
                    print(f"Error stopping player AI: {e}")
            if hasattr(self, "opponent") and self.opponent.ai_controller:
                try:
                    if hasattr(self.opponent.ai_controller, "_save_best_agent"):
                        self.opponent.ai_controller._save_best_agent()
                except Exception as e:
                    print(f"Error saving opponent AI state: {e}")
            if hasattr(self, "player") and self.player.ai_controller:
                try:
                    if hasattr(self.player.ai_controller, "_save_best_agent"):
                        self.player.ai_controller._save_best_agent()
                except Exception as e:
                    print(f"Error saving player AI state: {e}")
            self._resources_initialized = False
            print("Game cleanup completed successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")
            self._resources_initialized = False

    def __del__(self):
        """
        Cleanup the game when the object is deleted.
        Returns:
            None
        """
        if self._resources_initialized:
            self.cleanup()

    def set_two_player_mode(self, enabled=True):
        """
        Set the two player mode.
        Args:
            enabled (bool): Whether to enable two player mode
        Returns:
            None
        """
        self.two_player_mode = enabled
        if enabled:
            self.opponent.is_ai = False
            if self.opponent.ai_controller:
                self.opponent.ai_controller.stop_training()
                self.opponent.ai_controller = None
            print("Two-player mode enabled")
            print("Controls: Player 1 - arrows ↑↓, Player 2 - W/S")
        else:
            self.opponent.is_ai = True
            print("Single-player mode enabled")

    def handle_pause_menu_action(self, action):
        """
        Handle a pause menu action.
        Args:
            action (str): The action to handle
        Returns:
            None
        """
        if action == "Continue":
            self.pause_menu_active = False
            self.paused = False
        elif action == "Save AI":
            if self.opponent.ai_controller:
                if hasattr(self.opponent.ai_controller, "_save_best_agent"):
                    self.opponent.ai_controller._save_best_agent()
                    print("AI state saved successfully")
                else:
                    print("Save function not available")
        elif action == "Reset AI":
            if self.opponent.ai_controller:
                self.opponent.ai_controller.reset_training()
                print("AI training reset")
        elif action == "Settings":
            self.show_metrics = not self.show_metrics
            print(f"Metrics display: {'enabled' if self.show_metrics else 'disabled'}")
        elif action == "Exit to Menu":
            self.return_to_menu = True

    def handle_events(self):
        """
        Handle events.
        Returns:
            None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.save_and_exit()
            if self.pause_menu_active:
                action = self.pause_menu.handle_events(event)
                if action:
                    self.handle_pause_menu_action(action)
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.two_player_mode or self.training_mode:
                        self.return_to_menu = True
                    else:
                        self.pause_menu_active = True
                        self.paused = True
                if not self.paused:
                    if event.key == pygame.K_n:
                        self.use_neon_theme = not self.use_neon_theme
                        theme_status = "enabled" if self.use_neon_theme else "disabled"
                        print(f"Neon theme: {theme_status}")
                    elif event.key == pygame.K_r:
                        if self.opponent.ai_controller:
                            self.opponent.ai_controller.reset_training()
                            print("AI training reset")
                    elif event.key == pygame.K_s:
                        if self.opponent.ai_controller:
                            if hasattr(self.opponent.ai_controller, "_save_best_agent"):
                                self.opponent.ai_controller._save_best_agent()
                                print("Best AI state saved")
                    elif event.key == pygame.K_f:
                        self.fast_training = not self.fast_training
                        print(
                            f"Fast training: {'enabled' if self.fast_training else 'disabled'}"
                        )
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.simulation_speed = min(1000.0, self.simulation_speed * 1.5)
                        print(f"Simulation speed: {self.simulation_speed:.1f}x")
                    elif event.key == pygame.K_MINUS:
                        self.simulation_speed = max(0.1, self.simulation_speed / 1.5)
                        print(f"Simulation speed: {self.simulation_speed:.1f}x")
                    elif event.key == pygame.K_m:
                        self.show_metrics = not self.show_metrics
                        print(f"Metrics: {'shown' if self.show_metrics else 'hidden'}")
                    elif event.key == pygame.K_v:
                        self.show_nn_visualization = not self.show_nn_visualization
                        self.nn_visualizer_manager.set_all_enabled(
                            self.show_nn_visualization
                        )
                        print(
                            f"Neural Network Visualization: {'enabled' if self.show_nn_visualization else 'disabled'}"
                        )
                    elif event.key == pygame.K_t and self.training_mode:
                        self.fast_training = not self.fast_training
                        if self.fast_training:
                            print("Fast training enabled (10x speed)")
                            self.simulation_speed = 10.0
                        else:
                            print("Normal training speed (1x speed)")
                            self.simulation_speed = 1.0
                    elif not self.training_mode:
                        if event.key == pygame.K_1:
                            if self.powerup_manager.activate_powerup(
                                "player", "size_boost"
                            ):
                                print("Activated power-up: player paddle size boost")
                        elif event.key == pygame.K_2:
                            if self.powerup_manager.activate_powerup(
                                "player", "speed_boost"
                            ):
                                print("Activated power-up: player paddle speed boost")
                        elif event.key == pygame.K_3:
                            if self.powerup_manager.activate_powerup(
                                "ball", "slowdown"
                            ):
                                print("Activated power-up: ball slowdown")
                        elif event.key == pygame.K_4:
                            if self.powerup_manager.activate_powerup(
                                "opponent", "speed_boost"
                            ):
                                print("Activated power-up: AI speed boost")
                        elif event.key == pygame.K_5:
                            if self.powerup_manager.activate_powerup(
                                "opponent", "size_boost"
                            ):
                                print("Activated power-up: AI paddle size boost")
                        elif event.key == pygame.K_6:
                            if self.powerup_manager.activate_powerup(
                                "ball", "multiball"
                            ):
                                print("Activated power-up: multiball!")
                        elif event.key == pygame.K_7:
                            if self.powerup_manager.activate_powerup("ball", "chaotic"):
                                print("Activated power-up: chaotic ball!")
                        elif event.key == pygame.K_0:
                            self.powerup_manager.combo_count = 5
                            self.powerup_manager.combo_multiplier = 2.0
                            print("Forced combo x5 for testing")

    def save_and_exit(self):
        """
        Save the AI state and exit the game.
        Returns:
            None
        """
        if self.opponent.ai_controller:
            print("\nSaving AI state before exit...")
            self.opponent.ai_controller.stop_training()
            if hasattr(self.opponent.ai_controller, "_save_best_agent"):
                self.opponent.ai_controller._save_best_agent()
            print("State saved")
        self.return_to_menu = True

    def reset_game(self):
        """
        Reset the game.
        Returns:
            None
        """
        try:
            new_ball = Ball(
                self.WIDTH // 2 - self.BALL_SIZE // 2,
                self.HEIGHT // 2 - self.BALL_SIZE // 2,
                self.BALL_SIZE,
                self.BALL_SPEED_X,
                self.BALL_SPEED_Y,
            )
            new_ball.reset(self.WIDTH // 2, self.HEIGHT // 2)
            self.balls = [new_ball]
            self.ball = self.balls[0]
            self.round_goals = 0
            self.player.rect.y = self.HEIGHT // 2 - self.player.rect.height // 2
            self.opponent.rect.y = self.HEIGHT // 2 - self.opponent.rect.height // 2
            if self.player.ai_controller:
                self.player.ai_controller.reset_round_metrics()
            if self.opponent.ai_controller:
                self.opponent.ai_controller.reset_round_metrics()
            if (
                self.training_mode
                and self.player.ai_controller
                and self.opponent.ai_controller
            ):
                if hasattr(self.player.ai_controller, "_exchange_experience"):
                    self.player.ai_controller._exchange_experience()
                if hasattr(self.opponent.ai_controller, "_exchange_experience"):
                    self.opponent.ai_controller._exchange_experience()
                print(f"Round {self.rounds_completed} completed - weights exchanged")
        except Exception as e:
            print(f"Error resetting game: {e}")
            self.return_to_menu = True

    def _update_main_ball_reference(self):
        """
        Update the main ball reference.
        Returns:
            None
        """
        if not self.balls:
            return
        closest_ball = min(
            self.balls,
            key=lambda ball: abs(ball.rect.centerx - self.opponent.rect.centerx)
            + abs(ball.rect.centery - self.opponent.rect.centery),
        )
        self.ball = closest_ball
        if self.opponent.ai_controller:
            self.opponent.ai_controller.ball = closest_ball
            if hasattr(self.opponent.ai_controller, "agents"):
                for agent in self.opponent.ai_controller.agents:
                    agent.ball = closest_ball
            if (
                hasattr(self.opponent.ai_controller, "best_agent")
                and self.opponent.ai_controller.best_agent
            ):
                self.opponent.ai_controller.best_agent.ball = closest_ball
        if self.player.ai_controller:
            self.player.ai_controller.ball = closest_ball
            if hasattr(self.player.ai_controller, "agents"):
                for agent in self.player.ai_controller.agents:
                    agent.ball = closest_ball
            if (
                hasattr(self.player.ai_controller, "best_agent")
                and self.player.ai_controller.best_agent
            ):
                self.player.ai_controller.best_agent.ball = closest_ball

    def update(self):
        """
        Update the game.
        Returns:
            None
        """
        with self.state_lock:
            if self.paused:
                return
            try:
                self.player.update()
                self.opponent.update()
                balls_to_remove = []
                player_hit = False
                opponent_hit = False
                self._update_main_ball_reference()
                for i, ball in enumerate(self.balls):
                    ball.move()
                    ball.bounce(0, self.HEIGHT, 0, self.WIDTH)
                    if ball.rect.colliderect(self.player.rect) or (
                        ball.speed_x < 0
                        and ball.rect.left <= self.player.rect.right
                        and ball.rect.right >= self.player.rect.left
                        and ball.rect.centery >= self.player.rect.top
                        and ball.rect.centery <= self.player.rect.bottom
                    ):
                        if ball.handle_paddle_collision(self.player):
                            player_hit = True
                    if ball.rect.colliderect(self.opponent.rect) or (
                        ball.speed_x > 0
                        and ball.rect.right >= self.opponent.rect.left
                        and ball.rect.left <= self.opponent.rect.right
                        and ball.rect.centery >= self.opponent.rect.top
                        and ball.rect.centery <= self.opponent.rect.bottom
                    ):
                        if ball.handle_paddle_collision(self.opponent):
                            opponent_hit = True
                    if ball.rect.left <= 0:
                        self.score.update(False, True)
                        self.round_goals += 1
                        balls_to_remove.append(i)
                        if not self.player.is_ai:
                            self.powerup_manager.update_combo(reset_player1=True)
                    elif ball.rect.right >= self.WIDTH:
                        self.score.update(True, False)
                        self.round_goals += 1
                        balls_to_remove.append(i)
                        if self.two_player_mode and not self.opponent.is_ai:
                            self.powerup_manager.update_combo(reset_player2=True)
                for i in reversed(balls_to_remove):
                    del self.balls[i]
                if not self.balls:
                    new_ball = Ball(
                        self.WIDTH // 2 - self.BALL_SIZE // 2,
                        self.HEIGHT // 2 - self.BALL_SIZE // 2,
                        self.BALL_SIZE,
                        self.BALL_SPEED_X,
                        self.BALL_SPEED_Y,
                    )
                    new_ball.reset(self.WIDTH // 2, self.HEIGHT // 2)
                    self.balls.append(new_ball)
                    self.ball = self.balls[0]
                if player_hit:
                    if not self.player.is_ai:
                        self.powerup_manager.update_combo(player_hit=True)
                        if self.powerup_manager.player1_combo_count >= 3:
                            self.powerup_manager.try_spawn_powerup("combo")
                        else:
                            self.powerup_manager.try_spawn_powerup("hit")
                    if self.player.ai_controller:
                        self.player.ai_controller.notify_game_event("paddle_hit")
                    if (
                        self.player.ai_controller
                        and not self.two_player_mode
                        and self.training_mode
                    ):
                        try:
                            if hasattr(self.player.ai_controller, "train"):
                                self.player.ai_controller.train()
                        except Exception as e:
                            print(f"Error during player AI training: {e}")
                if opponent_hit:
                    self.powerup_manager.update_combo(player2_hit=True)
                    if self.powerup_manager.player2_combo_count >= 3:
                        self.powerup_manager.try_spawn_powerup("combo")
                    else:
                        self.powerup_manager.try_spawn_powerup("hit")
                    if self.opponent.ai_controller:
                        self.opponent.ai_controller.notify_game_event("paddle_hit")
                    if (
                        self.opponent.ai_controller
                        and not self.two_player_mode
                        and self.training_mode
                    ):
                        try:
                            if hasattr(self.opponent.ai_controller, "train"):
                                self.opponent.ai_controller.train()
                        except Exception as e:
                            print(f"Error during opponent AI training: {e}")
                if (
                    self.training_mode
                    and self.player.ai_controller
                    and self.opponent.ai_controller
                    and balls_to_remove
                ):
                    if hasattr(self.player.ai_controller, "_update_agent_scores"):
                        player_goals = len(
                            [
                                i
                                for i in balls_to_remove
                                if i < len(self.balls)
                                and self.balls[i].rect.right >= self.WIDTH
                            ]
                        )
                        opponent_goals = len(
                            [
                                i
                                for i in balls_to_remove
                                if i < len(self.balls) and self.balls[i].rect.left <= 0
                            ]
                        )
                        if player_goals > opponent_goals:
                            self.player.ai_controller._update_agent_scores(
                                self.player.ai_controller.get_active_agent().agent_id,
                                self.opponent.ai_controller.get_active_agent().agent_id,
                            )
                        elif opponent_goals > player_goals:
                            self.player.ai_controller._update_agent_scores(
                                self.opponent.ai_controller.get_active_agent().agent_id,
                                self.player.ai_controller.get_active_agent().agent_id,
                            )
                if self.round_goals >= self.goals_per_round:
                    if self.player.ai_controller:
                        self.player.ai_controller.notify_game_event("round_end")
                    if self.opponent.ai_controller:
                        self.opponent.ai_controller.notify_game_event("round_end")
                    self.rounds_completed += 1
                    self.reset_game()
                if not self.two_player_mode and self.training_mode:
                    if self.player.ai_controller:
                        try:
                            if hasattr(self.player.ai_controller, "train"):
                                self.player.ai_controller.train()
                        except Exception as e:
                            print(f"Error during player AI training: {e}")
                    if self.opponent.ai_controller:
                        try:
                            if hasattr(self.opponent.ai_controller, "train"):
                                self.opponent.ai_controller.train()
                        except Exception as e:
                            print(f"Error during opponent AI training: {e}")
                self.powerup_manager.update()
                if self.powerup_manager.ball_powerups["chaotic"]["active"]:
                    self.powerup_manager._update_chaotic_ball()
                if self.show_nn_visualization:
                    neural_networks = {}
                    if self.opponent.is_ai and self.opponent.ai_controller:
                        active_agent = self.opponent.ai_controller.get_active_agent()
                        if active_agent and hasattr(active_agent, "neural_network"):
                            neural_networks["opponent"] = active_agent.neural_network
                    if self.player.is_ai and self.player.ai_controller:
                        active_agent = self.player.ai_controller.get_active_agent()
                        if active_agent and hasattr(active_agent, "neural_network"):
                            neural_networks["player"] = active_agent.neural_network
                    self.nn_visualizer_manager.update(neural_networks)
                    if (
                        hasattr(self, "_debug_frame_count")
                        and self._debug_frame_count % 300 == 0
                    ):
                        print(
                            f"NN Visualization active with {len(neural_networks)} networks"
                        )
            except Exception as e:
                print(f"Error in game update: {e}")
                self.return_to_menu = True

    def draw(self):
        """
        Draw the game.
        Returns:
            None
        """
        with self.state_lock:
            if not hasattr(self, "_debug_frame_count"):
                self._debug_frame_count = 0
            self._debug_frame_count += 1
            current_time = time.time()
            if not hasattr(self, "last_frame_time") or self.last_frame_time == 0:
                self.last_frame_time = current_time
                dt = 1.0 / 60.0
            else:
                dt = current_time - self.last_frame_time
            dt = min(dt, 1.0 / 60.0)
            if self.use_neon_theme:
                self.neon_theme.update_animations(dt)
            if self.use_neon_theme:
                self.neon_theme.draw_gradient_background(
                    self.screen, self.WIDTH, self.HEIGHT
                )
            else:
                self.screen.fill(self.BLACK)
            if self.use_neon_theme:
                player_color = self.neon_theme.NEON_CYAN
                opponent_color = self.neon_theme.NEON_PINK
                if not hasattr(self.player, "_last_y"):
                    self.player._last_y = self.player.rect.centery
                if not hasattr(self.opponent, "_last_y"):
                    self.opponent._last_y = self.opponent.rect.centery
                player_moved = abs(self.player._last_y - self.player.rect.centery) > 3
                opponent_moved = (
                    abs(self.opponent._last_y - self.opponent.rect.centery) > 3
                )
                if player_moved:
                    self.neon_theme.add_paddle_particle(
                        self.player.rect.centerx, self.player.rect.centery, player_color
                    )
                if opponent_moved:
                    self.neon_theme.add_paddle_particle(
                        self.opponent.rect.centerx,
                        self.opponent.rect.centery,
                        opponent_color,
                    )
                self.player._last_y = self.player.rect.centery
                self.opponent._last_y = self.opponent.rect.centery
                self.neon_theme.draw_glow_rect(
                    self.screen, player_color, self.player.rect
                )
                self.neon_theme.draw_glow_rect(
                    self.screen, opponent_color, self.opponent.rect
                )
            else:
                self.player.draw(self.screen, self.WHITE)
                self.opponent.draw(self.screen, self.WHITE)
            for i, ball in enumerate(self.balls):
                if self.use_neon_theme:
                    chaos_color = self.powerup_manager.get_chaos_color()
                    if chaos_color:
                        ball_color = chaos_color
                    elif ball == self.ball:
                        ball_color = self.neon_theme.NEON_YELLOW
                    else:
                        ball_color = self.neon_theme.NEON_GREEN
                    self.neon_theme.add_ball_particle(
                        ball.rect.centerx, ball.rect.centery, ball_color
                    )
                    self.neon_theme.draw_glow_circle(
                        self.screen,
                        ball_color,
                        (ball.rect.centerx, ball.rect.centery),
                        ball.rect.width // 2,
                    )
                else:
                    chaos_color = self.powerup_manager.get_chaos_color()
                    if chaos_color:
                        ball.draw(self.screen, chaos_color)
                    elif ball == self.ball:
                        ball.draw(self.screen, self.WHITE)
                    else:
                        ball.draw(self.screen, (200, 200, 200))
            if self.use_neon_theme:
                center_x = self.WIDTH // 2
                line_color = self.neon_theme.NEON_PURPLE
                self.neon_theme.draw_glow_line(
                    self.screen,
                    line_color,
                    (center_x, 0),
                    (center_x, self.HEIGHT),
                    width=2,
                    glow_size=6,
                )
            else:
                pygame.draw.aaline(
                    self.screen,
                    self.WHITE,
                    (self.WIDTH // 2, 0),
                    (self.WIDTH // 2, self.HEIGHT),
                )
            if self.use_neon_theme:
                score_color = self.neon_theme.NEON_ORANGE
                font = pygame.font.Font(None, 36)
                player_score_text = str(self.score.player_score)
                self.neon_theme.draw_neon_text(
                    self.screen,
                    font,
                    player_score_text,
                    score_color,
                    (self.WIDTH // 4 - 20, 50),
                )
                opponent_score_text = str(self.score.opponent_score)
                self.neon_theme.draw_neon_text(
                    self.screen,
                    font,
                    opponent_score_text,
                    score_color,
                    (3 * self.WIDTH // 4 - 20, 50),
                )
            else:
                self.score.draw(self.screen, self.WHITE, self.WIDTH)
            if self.use_neon_theme:
                self.neon_theme.draw_particles(self.screen)
            if self.training_mode:
                round_text = f"Round: {self.rounds_completed} | Goals: {self.round_goals}/{self.goals_per_round}"
                font = pygame.font.Font(None, 24)
                if self.use_neon_theme:
                    self.neon_theme.draw_neon_text(
                        self.screen,
                        font,
                        round_text,
                        self.neon_theme.NEON_BLUE,
                        (self.WIDTH // 2 - 100, 50),
                    )
                else:
                    text_surface = font.render(round_text, True, self.WHITE)
                    self.screen.blit(text_surface, (self.WIDTH // 2 - 100, 50))
            if self.show_metrics:
                self.metrics_display.draw(self.screen)
            if self.show_nn_visualization:
                self.nn_visualizer_manager.draw(self.screen)
            if self.pause_menu_active:
                self.pause_menu.draw()
            if self.training_mode:
                hint_text = (
                    "T-speed | F-fast | M-metrics | N-neon | V-neural | ESC-pause"
                )
                font = pygame.font.Font(None, 24)
                if self.use_neon_theme:
                    self.neon_theme.draw_neon_text(
                        self.screen,
                        font,
                        hint_text,
                        self.neon_theme.NEON_GREEN,
                        (10, self.HEIGHT - 30),
                    )
                else:
                    text_surface = font.render(hint_text, True, self.WHITE)
                    self.screen.blit(text_surface, (10, self.HEIGHT - 30))
            elif not self.training_mode:
                if self.two_player_mode:
                    hint_text = "1-size | 2-speed | 3-slow | 6-multiball | 7-chaos | N-neon | PvP"
                else:
                    hint_text = "1-size | 2-speed | 3-slow | 6-multiball | 7-chaos | N-neon | ESC-pause"
                font = pygame.font.Font(None, 16)
                if self.use_neon_theme:
                    self.neon_theme.draw_neon_text(
                        self.screen,
                        font,
                        hint_text,
                        self.neon_theme.NEON_CYAN,
                        (10, self.HEIGHT - 25),
                    )
                else:
                    text_surface = font.render(hint_text, True, self.WHITE)
                    self.screen.blit(text_surface, (10, self.HEIGHT - 25))
            self.powerup_manager.draw_powerup_indicators(self.screen)
            self.last_frame_time = time.time()
            pygame.display.flip()

    def run(self):
        """
        Run the game.
        Returns:
            None
        """
        if not self._resources_initialized:
            print("Game resources not properly initialized")
            return
        clock = pygame.time.Clock()
        try:
            while True:
                if getattr(self, "return_to_menu", False):
                    print("Returning to menu...")
                    break
                current_time = time.time()
                delta_time = current_time - self.last_frame_time
                self.handle_events()
                if getattr(self, "return_to_menu", False):
                    print("Return to menu requested")
                    break
                if not self.paused:
                    for _ in range(int(self.simulation_speed)):
                        self.update()
                should_render = True
                if should_render:
                    self.draw()
                    if not self.fast_training and delta_time < self.frame_time:
                        time.sleep(self.frame_time - delta_time)
                else:
                    time.sleep(0.001)
                if not self.fast_training:
                    clock.tick(self.FPS)
        except KeyboardInterrupt:
            print("\nGame interrupted...")
        except Exception as e:
            print(f"\nGame loop error: {e}")
        finally:
            print("Game loop ended, starting cleanup...")
            self.cleanup()

    def _notify_ai_hit(self, paddle, is_successful_hit=True):
        """
        Notify the AI of a hit.
        Args:
            paddle (Paddle): The paddle that was hit
            is_successful_hit (bool): Whether the hit was successful
        Returns:
            None
        """
        if paddle.ai_controller and hasattr(paddle.ai_controller, "notify_hit_event"):
            paddle.ai_controller.notify_hit_event(is_successful_hit)

    def _notify_ai_goal(self, scoring_paddle, conceding_paddle):
        """
        Notify the AI of a goal.
        Args:
            scoring_paddle (Paddle): The paddle that scored
            conceding_paddle (Paddle): The paddle that conceded
        Returns:
            None
        """
        if scoring_paddle.ai_controller and hasattr(
            scoring_paddle.ai_controller, "notify_goal_event"
        ):
            scoring_paddle.ai_controller.notify_goal_event(scored=True)
        if conceding_paddle.ai_controller and hasattr(
            conceding_paddle.ai_controller, "notify_goal_event"
        ):
            conceding_paddle.ai_controller.notify_goal_event(scored=False)


if __name__ == "__main__":
    game = Game()
    game.run()
