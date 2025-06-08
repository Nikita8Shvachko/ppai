from enum import Enum
from typing import Dict

import pygame

from ai_difficulty_presets import AIDifficultyPresets

try:
    from pong_game import NeonTheme
except ImportError:
    NeonTheme = None


class GameState(Enum):
    """
    Enum for the game states.
    """

    MAIN_MENU = "main_menu"
    TRAINING_MENU = "training_menu"
    PLAY_MENU = "play_menu"
    DIFFICULTY_MENU = "difficulty_menu"
    AI_TRAINING_MENU = "ai_training_menu"
    TRAINING_SETUP_MENU = "training_setup_menu"
    MODEL_VIEW_MENU = "model_view_menu"
    TRAINING_GAME = "training_game"
    PLAYING_GAME = "playing_game"
    SETTINGS_MENU = "settings_menu"
    ABOUT_MENU = "about_menu"
    ERROR_STATE = "error_state"


class AIDifficultySelection:
    """
    AIDifficultySelection class.
    """

    @staticmethod
    def get_difficulty_options():
        """
        Get the difficulty options.
        Returns:
            List[Dict]: The difficulty options
        """
        presets = AIDifficultyPresets.get_all_presets()
        return [
            {
                "name": preset["name"],
                "description": preset["description"],
                "preset_data": preset,
            }
            for preset in presets
        ]


class MenuManager:
    """
    Menu manager for the game.
    """

    def __init__(self, screen_width=1200, screen_height=900):
        """
        Initialize the menu manager.
        Args:
            screen_width: The width of the screen
            screen_height: The height of the screen
        Returns:
            None
        """
        try:
            pygame.init()
            self.width = screen_width
            self.height = screen_height
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Pong AI")

            self.WHITE = (255, 255, 255)
            self.BLACK = (0, 0, 0)
            self.BLUE = (0, 100, 200)
            self.LIGHT_BLUE = (100, 150, 255)
            self.GREEN = (0, 200, 0)
            self.RED = (200, 0, 0)
            self.GRAY = (128, 128, 128)

            self.font_large = pygame.font.Font(None, 56)
            self.font_medium = pygame.font.Font(None, 40)
            self.font_small = pygame.font.Font(None, 32)
            self.current_state = GameState.MAIN_MENU
            self.selected_difficulty = None
            self.selected_model_file = None
            self.available_models = []
            self.selected_training_difficulty = None
            self.selected_training_mode = None
            self.two_player_mode = False
            self.clock = pygame.time.Clock()
            self.running = True
            self._resources_initialized = True

            self.menu_items = self._initialize_menus()

            self.selected_index = 0

            self.training_active = False
            self.game_instance = None

            self.neon_theme = NeonTheme() if NeonTheme else None
            self.use_neon_theme = True
            self._add_neon_toggle_to_menus()

        except Exception as e:
            print(f"Error initializing menu manager: {e}")
            self._resources_initialized = False
            self.cleanup()
            raise

    def cleanup(self):
        """
        Clean up the menu manager.
        Returns:
            None
        """
        try:
            if hasattr(self, "game_instance") and self.game_instance:
                try:
                    self.game_instance.cleanup()
                except Exception as e:
                    print(f"Error cleaning up game instance: {e}")

        except Exception as e:
            print(f"Error during menu cleanup: {e}")

    def __del__(self):
        """
        Destructor for the menu manager.
        Returns:
            None
        """
        if self._resources_initialized:
            self.cleanup()

    def _handle_state_change(self, new_state: GameState):
        """
        Handle the state change.
        Args:
            new_state: The new state
        Returns:
            None
        """
        try:
            if not self._is_valid_state_transition(self.current_state, new_state):
                print(
                    f"Invalid state transition: {self.current_state} -> " f"{new_state}"
                )
                self.current_state = GameState.ERROR_STATE
                return

            old_state = self.current_state
            self.current_state = new_state
            self.selected_index = 0

            if old_state in [GameState.TRAINING_GAME, GameState.PLAYING_GAME]:
                if self.game_instance:
                    self.game_instance.cleanup()
                    self.game_instance = None

            print(f"State changed: {old_state} -> {new_state}")

        except Exception as e:
            print(f"Error changing state: {e}")
            self.current_state = GameState.ERROR_STATE
            self.selected_index = 0

    def _is_valid_state_transition(self, current: GameState, new: GameState) -> bool:
        """
        Check if the state transition is valid.
        Args:
            current: The current state
            new: The new state
        Returns:
            bool: True if the state transition is valid, False otherwise
        """
        valid_transitions = {
            GameState.MAIN_MENU: [
                GameState.PLAY_MENU,
                GameState.TRAINING_MENU,
                GameState.AI_TRAINING_MENU,
                GameState.SETTINGS_MENU,
                GameState.ABOUT_MENU,
                GameState.ERROR_STATE,
            ],
            GameState.PLAY_MENU: [
                GameState.DIFFICULTY_MENU,
                GameState.PLAYING_GAME,
                GameState.MAIN_MENU,
                GameState.ERROR_STATE,
            ],
            GameState.DIFFICULTY_MENU: [
                GameState.PLAYING_GAME,
                GameState.PLAY_MENU,
                GameState.ERROR_STATE,
            ],
            GameState.TRAINING_MENU: [
                GameState.TRAINING_GAME,
                GameState.MAIN_MENU,
                GameState.ERROR_STATE,
            ],
            GameState.AI_TRAINING_MENU: [
                GameState.TRAINING_SETUP_MENU,
                GameState.MODEL_VIEW_MENU,
                GameState.MAIN_MENU,
                GameState.ERROR_STATE,
            ],
            GameState.TRAINING_SETUP_MENU: [
                GameState.TRAINING_GAME,
                GameState.AI_TRAINING_MENU,
                GameState.ERROR_STATE,
            ],
            GameState.TRAINING_GAME: [
                GameState.TRAINING_MENU,
                GameState.AI_TRAINING_MENU,
                GameState.MAIN_MENU,
                GameState.ERROR_STATE,
            ],
            GameState.PLAYING_GAME: [
                GameState.PLAY_MENU,
                GameState.MAIN_MENU,
                GameState.ERROR_STATE,
            ],
            GameState.MODEL_VIEW_MENU: [
                GameState.AI_TRAINING_MENU,
                GameState.MAIN_MENU,
                GameState.ERROR_STATE,
            ],
            GameState.SETTINGS_MENU: [GameState.MAIN_MENU, GameState.ERROR_STATE],
            GameState.ABOUT_MENU: [GameState.MAIN_MENU, GameState.ERROR_STATE],
            GameState.ERROR_STATE: [
                GameState.MAIN_MENU,
                GameState.PLAY_MENU,
                GameState.TRAINING_MENU,
                GameState.AI_TRAINING_MENU,
            ],
        }
        return new in valid_transitions.get(current, [])

    def _go_to_main_menu(self):
        """
        Go to the main menu.
        Returns:
            None
        """
        self._handle_state_change(GameState.MAIN_MENU)

    def _go_to_play_menu(self):
        """
        Go to the play menu.
        Returns:
            None
        """
        self._handle_state_change(GameState.PLAY_MENU)

    def _go_to_training_menu(self):
        """
        Go to the training menu.
        Returns:
            None
        """
        self._handle_state_change(GameState.TRAINING_MENU)

    def _go_to_ai_training(self):
        """
        Go to the AI training menu.
        Returns:
            None
        """
        self._refresh_training_menus()
        self._handle_state_change(GameState.AI_TRAINING_MENU)

    def _go_to_training_setup(self):
        """
        Go to the training setup menu.
        Returns:
            None
        """
        self._refresh_training_setup_menu()
        self._handle_state_change(GameState.TRAINING_SETUP_MENU)

    def _go_to_model_view(self):
        """
        Go to the model view menu.
        Returns:
            None
        """
        self._refresh_model_view_menu()
        self._handle_state_change(GameState.MODEL_VIEW_MENU)

    def _go_to_difficulty(self):
        """
        Go to the difficulty menu.
        Returns:
            None
        """
        self._refresh_difficulty_menu()
        self._handle_state_change(GameState.DIFFICULTY_MENU)

    def _go_to_settings(self):
        """
        Go to the settings menu.
        Returns:
            None
        """
        self._handle_state_change(GameState.SETTINGS_MENU)

    def _go_to_about(self):
        """
        Go to the about menu.
        Returns:
            None
        """
        self._handle_state_change(GameState.ABOUT_MENU)

    def _start_game_with_difficulty(self, difficulty_preset):
        """
        Start a game with a difficulty preset.
        Args:
            difficulty_preset: The difficulty preset
        Returns:
            None
        """
        try:
            self.selected_difficulty = difficulty_preset
            self.two_player_mode = False
            self._handle_state_change(GameState.PLAYING_GAME)
        except Exception as e:
            print(f"Error starting game with difficulty: {e}")
            self._go_to_play_menu()

    def _play_vs_best(self):
        """
        Play a game against the best model.
        Returns:
            None
        """
        try:
            self.selected_difficulty = None
            self.two_player_mode = False
            self._handle_state_change(GameState.PLAYING_GAME)
        except Exception as e:
            print(f"Error starting game vs best model: {e}")
            self._go_to_play_menu()

    def _play_two_players(self):
        """
        Play a two player game.
        Returns:
            None
        """
        try:
            self.selected_difficulty = None
            self.two_player_mode = True
            self._handle_state_change(GameState.PLAYING_GAME)
        except Exception as e:
            print(f"Error starting two player game: {e}")
            self._go_to_play_menu()

    def _quit_game(self):
        """
        Quit the game.
        Returns:
            None
        """
        try:
            self.cleanup()
            self.running = False
        except Exception as e:
            print(f"Error quitting game: {e}")
            self.running = False

    def _initialize_menus(self) -> Dict[GameState, list]:
        """
        Initialize the menus.
        Returns:
            Dict[GameState, list]: The menus
        """
        return {
            GameState.MAIN_MENU: [
                {"text": "Play", "action": self._go_to_play_menu},
                {"text": "Quick AI Training", "action": self._go_to_training_menu},
                {"text": "Deep AI Training", "action": self._go_to_ai_training},
                {"text": "Settings", "action": self._go_to_settings},
                {"text": "About", "action": self._go_to_about},
                {"text": "Exit", "action": self._quit_game},
            ],
            GameState.PLAY_MENU: [
                {"text": "Select AI Difficulty", "action": self._go_to_difficulty},
                {
                    "text": "Play Against Best Model",
                    "action": self._play_vs_best,
                },
                {"text": "Two Player Game", "action": self._play_two_players},
                {"text": "Back", "action": self._go_to_main_menu},
            ],
            GameState.DIFFICULTY_MENU: [],
            GameState.TRAINING_MENU: [
                {"text": "Start Training", "action": self._start_training},
                {"text": "Continue Training", "action": self._continue_training},
                {"text": "Reset Model", "action": self._reset_model},
                {"text": "Quick Training", "action": self._start_fast_training},
                {"text": "View Statistics", "action": self._view_statistics},
                {"text": "Back", "action": self._go_to_main_menu},
            ],
            GameState.AI_TRAINING_MENU: [],
            GameState.TRAINING_SETUP_MENU: [],
            GameState.MODEL_VIEW_MENU: [],
            GameState.SETTINGS_MENU: [
                {
                    "text": "Game Speed: Normal",
                    "action": self._toggle_game_speed,
                },
                {"text": "Sound: Enabled", "action": self._toggle_sound},
                {"text": "Show FPS: Disabled", "action": self._toggle_fps},
                {"text": "Reset All Settings", "action": self._reset_settings},
                {"text": "Back", "action": self._go_to_main_menu},
            ],
            GameState.ABOUT_MENU: [
                {"text": "Version: 2.0", "action": None},
                {"text": "AI with human limitations", "action": None},
                {"text": "Integrated training system", "action": None},
                {"text": "Deep learning: ParallelAI", "action": None},
                {"text": "Back", "action": self._go_to_main_menu},
            ],
        }

    def handle_events(self):
        """
        Handle the events.
        Returns:
            None
        """
        try:
            if not pygame.get_init():
                self.running = False
                return

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._handle_escape()

                    elif event.key == pygame.K_UP:
                        self._move_selection(-1)

                    elif event.key == pygame.K_DOWN:
                        self._move_selection(1)

                    elif event.key == pygame.K_RETURN or (event.key == pygame.K_SPACE):
                        self._execute_selected_action()

                    elif event.key == pygame.K_n:
                        self._toggle_neon_theme()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self._handle_mouse_click(event.pos)

                elif event.type == pygame.MOUSEMOTION:
                    self._handle_mouse_hover(event.pos)

        except pygame.error as e:
            print(f"Pygame error in event handling: {e}")
            self.running = False
        except Exception as e:
            print(f"Error handling events: {e}")
            self.current_state = GameState.ERROR_STATE

    def _handle_escape(self):
        """
        Handle the escape key.
        Returns:
            None
        """
        try:
            if self.current_state == GameState.MAIN_MENU:
                self.running = False
            elif self.current_state in [
                GameState.TRAINING_GAME,
                GameState.PLAYING_GAME,
            ]:
                self._go_to_main_menu()
            else:
                self._go_to_main_menu()
        except Exception as e:
            print(f"Error handling escape: {e}")
            self.running = False

    def _move_selection(self, direction: int):
        """
        Move the selection.
        Args:
            direction: The direction to move
        Returns:
            None
        """
        current_menu = self.menu_items.get(self.current_state, [])
        if not current_menu:
            return

        start_index = self.selected_index
        while True:
            self.selected_index = (self.selected_index + direction) % len(current_menu)
            item = current_menu[self.selected_index]

            if item.get("enabled", True) or self.selected_index == start_index:
                break

    def _execute_selected_action(self):
        """
        Execute the selected action.
        Returns:
            None
        """
        current_menu = self.menu_items.get(self.current_state, [])
        if current_menu and 0 <= self.selected_index < len(current_menu):
            item = current_menu[self.selected_index]
            action = item.get("action")
            is_enabled = item.get("enabled", True)
            if action and is_enabled:
                action()

    def _handle_mouse_click(self, pos):
        """
        Handle the mouse click.
        Args:
            pos: The position of the mouse
        Returns:
            None
        """
        current_menu = self.menu_items.get(self.current_state, [])
        for i, item in enumerate(current_menu):
            item_rect = self._get_menu_item_rect(i)
            if item_rect.collidepoint(pos):
                self.selected_index = i
                self._execute_selected_action()
                break

    def _handle_mouse_hover(self, pos):
        """
        Handle the mouse hover.
        Args:
            pos: The position of the mouse
        Returns:
            None
        """
        current_menu = self.menu_items.get(self.current_state, [])
        for i, item in enumerate(current_menu):
            item_rect = self._get_menu_item_rect(i)
            if item_rect.collidepoint(pos):
                self.selected_index = i
                break

    def _get_menu_item_rect(self, index: int) -> pygame.Rect:
        """
        Get the menu item rectangle.
        Args:
            index: The index of the menu item
        Returns:
            pygame.Rect: The menu item rectangle
        """
        current_menu = self.menu_items.get(self.current_state, [])
        if not current_menu:
            return pygame.Rect(0, 0, 0, 0)

        num_items = len(current_menu)
        if num_items > 12:
            item_spacing = 45
        elif num_items > 8:
            item_spacing = 50
        else:
            item_spacing = 60

        total_height = num_items * item_spacing
        start_y = max(150, self.height // 2 - total_height // 2)

        item_y = start_y + index * item_spacing

        return pygame.Rect(self.width // 2 - 300, item_y - 25, 600, 50)

    def _start_training(self):
        """
        Start AI training from scratch.
        Returns:
            None
        """
        print("Starting AI training from scratch")
        self.current_state = GameState.TRAINING_GAME
        self._start_pong_training(reset_model=True)

    def _continue_training(self):
        """
        Continue AI training.
        Returns:
            None
        """
        print("Continuing AI training")
        self.current_state = GameState.TRAINING_GAME
        self._start_pong_training(reset_model=False)

    def _start_fast_training(self):
        """
        Start quick training.
        Returns:
            None
        """
        print("Starting quick training")
        self.current_state = GameState.TRAINING_GAME
        self._start_pong_training(fast_mode=True)

    def _reset_model(self):
        """
        Reset the AI model.
        Returns:
            None
        """
        print("Resetting AI model")

    def _view_statistics(self):
        """
        View training statistics.
        Returns:
            None
        """
        print("Viewing training statistics")

    def _toggle_game_speed(self):
        """
        Toggle game speed.
        Returns:
            None
        """
        print("Toggling game speed")

    def _toggle_sound(self):
        """
        Toggle sound.
        Returns:
            None
        """
        print("Toggling sound")

    def _toggle_fps(self):
        """
        Toggle FPS.
        Returns:
            None
        """
        print("Toggling FPS")

    def _reset_settings(self):
        """
        Reset settings.
        Returns:
            None
        """
        print("Resetting settings")

    def _pause_or_return_to_menu(self):
        """
        Pause or return to menu.
        Returns:
            None
        """
        self._go_to_main_menu()

    def _start_pong_game(
        self, difficulty=None, use_best_model=False, two_players=False
    ):
        """
        Start a Pong game.
        Args:
            difficulty: The difficulty
            use_best_model: Whether to use the best model
            two_players: Whether to play two players
        Returns:
            None
        """
        model_file = getattr(self, "selected_model_file", None)

        print(
            f"Starting Pong game: difficulty={difficulty}, "
            f"best_model={use_best_model}, two_players={two_players}, "
            f"model_file={model_file}"
        )

        if hasattr(self, "_start_pong_game") and model_file:
            pass

    def _start_pong_training(self, reset_model=False, fast_mode=False):
        """
        Start Pong training.
        Args:
            reset_model: Whether to reset the model
            fast_mode: Whether to use fast mode
        Returns:
            None
        """
        print(f"Starting training: reset={reset_model}, " f"fast_mode={fast_mode}")

    def _toggle_neon_theme(self):
        """
        Toggle neon theme.
        Returns:
            None
        """
        if self.neon_theme:
            self.use_neon_theme = not self.use_neon_theme
            status = "enabled" if self.use_neon_theme else "disabled"
            print(f"Menu neon theme: {status}")

    def _add_neon_toggle_to_menus(self):
        """
        Add neon toggle to menus.
        Returns:
            None
        """
        pass

    def _draw_theme_hint(self):
        """
        Draw theme hint.
        Returns:
            None
        """
        hint_text = (
            f"Press 'N' to toggle neon theme ({'ON' if self.use_neon_theme else 'OFF'})"
        )
        hint_color = (
            self.neon_theme.NEON_GREEN if self.use_neon_theme else (128, 128, 128)
        )

        if self.use_neon_theme:
            self.neon_theme.draw_neon_text(
                self.screen,
                self.font_small,
                hint_text,
                hint_color,
                (10, self.height - 25),
            )
        else:
            hint_surface = self.font_small.render(hint_text, True, hint_color)
            self.screen.blit(hint_surface, (10, self.height - 25))

    def draw(self):
        """
        Draw the menu.
        Returns:
            None
        """
        if self.use_neon_theme and self.neon_theme:
            self.neon_theme.update_animations(self.clock.get_time() / 1000.0)
        try:
            if not pygame.get_init() or pygame.display.get_surface() is None:
                return

            if self.use_neon_theme and self.neon_theme:
                self.neon_theme.draw_gradient_background(
                    self.screen, self.width, self.height
                )
            else:
                self.screen.fill(self.BLACK)

            self._draw_title()

            self._draw_menu_items()

            self._draw_additional_info()

            if self.neon_theme:
                self._draw_theme_hint()

            pygame.display.flip()
        except pygame.error as e:
            print(f"Pygame display error: {e}")
            self.running = False
        except Exception as e:
            print(f"Error in draw method: {e}")
            self.running = False

    def _draw_title(self):
        """
        Draw the title.
        Returns:
            None
        """
        titles = {
            GameState.MAIN_MENU: "PONG AI",
            GameState.PLAY_MENU: "GAME MODE SELECTION",
            GameState.DIFFICULTY_MENU: "AI DIFFICULTY SELECTION",
            GameState.TRAINING_MENU: "QUICK TRAINING",
            GameState.AI_TRAINING_MENU: "DEEP AI TRAINING",
            GameState.TRAINING_SETUP_MENU: "TRAINING SETUP",
            GameState.MODEL_VIEW_MENU: "MODEL VIEWER",
            GameState.SETTINGS_MENU: "SETTINGS",
            GameState.ABOUT_MENU: "ABOUT",
            GameState.ERROR_STATE: "ERROR",
        }

        title_text = titles.get(self.current_state, "PONG AI")

        if self.use_neon_theme and self.neon_theme:
            title_color = self.neon_theme.NEON_CYAN
            self.neon_theme.draw_neon_text(
                self.screen,
                self.font_large,
                title_text,
                title_color,
                (self.width // 2 - self.font_large.size(title_text)[0] // 2, 80),
            )
        else:
            title_surface = self.font_large.render(title_text, True, self.WHITE)
            title_rect = title_surface.get_rect(center=(self.width // 2, 80))
            self.screen.blit(title_surface, title_rect)

    def _draw_menu_items(self):
        """
        Draw the menu items.
        Returns:
            None
        """
        current_menu = self.menu_items.get(self.current_state, [])
        if not current_menu:
            return

        num_items = len(current_menu)
        if num_items > 12:
            item_spacing = 45
        elif num_items > 8:
            item_spacing = 50
        else:
            item_spacing = 60

        total_height = num_items * item_spacing
        start_y = max(150, self.height // 2 - total_height // 2)

        for i, item in enumerate(current_menu):
            is_selected = i == self.selected_index
            is_enabled = item.get("enabled", True)

            if self.use_neon_theme and self.neon_theme:
                if not is_enabled:
                    color = (80, 80, 80)
                elif is_selected:
                    color = self.neon_theme.NEON_GREEN
                else:
                    color = self.neon_theme.NEON_BLUE
            else:
                if not is_enabled:
                    color = self.GRAY
                elif is_selected:
                    color = self.LIGHT_BLUE
                else:
                    color = self.WHITE

            text = item["text"]
            if (
                text.startswith("===")
                or text.startswith("Start Training")
                or text.startswith("Warning")
            ):
                font = self.font_medium
            elif is_selected:
                font = self.font_medium
            else:
                font = self.font_small

            if is_selected and is_enabled:
                item_rect = self._get_menu_item_rect(i)
                if self.use_neon_theme and self.neon_theme:
                    self.neon_theme.draw_glow_rect(
                        self.screen, self.neon_theme.NEON_PURPLE, item_rect, glow_size=8
                    )
                else:
                    pygame.draw.rect(self.screen, self.BLUE, item_rect, border_radius=8)

            text_pos = (self.width // 2, start_y + i * item_spacing)
            if self.use_neon_theme and self.neon_theme:
                text_x = text_pos[0] - font.size(text)[0] // 2
                self.neon_theme.draw_neon_text(
                    self.screen,
                    font,
                    text,
                    color,
                    (text_x, text_pos[1] - font.get_height() // 2),
                )
            else:
                text_surface = font.render(text, True, color)
                text_rect = text_surface.get_rect(center=text_pos)
                self.screen.blit(text_surface, text_rect)

    def _draw_additional_info(self):
        """
        Draw the additional info.
        Returns:
            None
        """
        info_y = self.height - 80

        if self.current_state == GameState.MAIN_MENU:
            info_text = "Use ↑↓ for navigation, Enter to select, Esc to exit"
            info_surface = self.font_small.render(info_text, True, self.GRAY)
            info_rect = info_surface.get_rect(center=(self.width // 2, info_y))
            self.screen.blit(info_surface, info_rect)

        elif self.current_state == GameState.DIFFICULTY_MENU:
            current_menu = self.menu_items.get(self.current_state, [])
            if current_menu and 0 <= self.selected_index < len(current_menu):
                selected_item = current_menu[self.selected_index]
                description = selected_item.get("description", "")
                if description:
                    if len(description) > 80:
                        words = description.split()
                        lines = []
                        current_line = []
                        current_length = 0

                        for word in words:
                            if current_length + len(word) + 1 <= 80:
                                current_line.append(word)
                                current_length += len(word) + 1
                            else:
                                if current_line:
                                    lines.append(" ".join(current_line))
                                current_line = [word]
                                current_length = len(word)

                        if current_line:
                            lines.append(" ".join(current_line))

                        for i, line in enumerate(lines):
                            desc_surface = self.font_small.render(
                                line, True, self.LIGHT_BLUE
                            )
                            desc_rect = desc_surface.get_rect(
                                center=(self.width // 2, info_y + i * 30)
                            )
                            self.screen.blit(desc_surface, desc_rect)
                    else:
                        desc_surface = self.font_small.render(
                            description, True, self.LIGHT_BLUE
                        )
                        desc_rect = desc_surface.get_rect(
                            center=(self.width // 2, info_y)
                        )
                        self.screen.blit(desc_surface, desc_rect)

        elif self.current_state in [
            GameState.TRAINING_SETUP_MENU,
            GameState.MODEL_VIEW_MENU,
        ]:
            current_menu = self.menu_items.get(self.current_state, [])
            if current_menu and 0 <= self.selected_index < len(current_menu):
                selected_item = current_menu[self.selected_index]
                description = selected_item.get("description", "")
                if description:
                    desc_surface = self.font_small.render(
                        description, True, self.LIGHT_BLUE
                    )
                    desc_rect = desc_surface.get_rect(center=(self.width // 2, info_y))
                    self.screen.blit(desc_surface, desc_rect)

        elif self.current_state == GameState.ABOUT_MENU:
            additional_info = [
                "Pong game with trainable AI",
                "AI with human limitations",
                "Deep learning with ParallelAIController",
                "Integrated difficulty system",
            ]

            start_y = self.height // 2 + 150
            for i, line in enumerate(additional_info):
                text_surface = self.font_small.render(line, True, self.GRAY)
                text_rect = text_surface.get_rect(
                    center=(self.width // 2, start_y + i * 35)
                )
                self.screen.blit(text_surface, text_rect)

    def run(self):
        """
        Run the menu.
        Returns:
            None
        """
        if not self._resources_initialized:
            print("Menu resources not properly initialized")
            return

        try:
            while self.running:
                self.handle_events()

                if self.current_state in [
                    GameState.TRAINING_GAME,
                    GameState.PLAYING_GAME,
                ]:
                    if self.current_state == GameState.TRAINING_GAME:
                        self._run_training_mode()
                    else:
                        self._run_playing_mode()
                else:
                    self.draw()
                    self.clock.tick(60)

        except Exception as e:
            print(f"Error in menu loop: {e}")
        finally:
            self.cleanup()

    def _run_training_mode(self):
        """
        Run the training mode.
        Returns:
            None
        """
        print("Training mode active")
        self._go_to_training_menu()

    def _run_playing_mode(self):
        """
        Run the playing mode.
        Returns:
            None
        """
        print("Game mode active")
        self._go_to_play_mode()

    def _refresh_training_menus(self):
        """
        Refresh the training menus.
        Returns:
            None
        """
        try:
            training_menu = [
                {"text": "Create New Model", "action": self._go_to_training_setup},
                {
                    "text": "View Trained Models",
                    "action": self._go_to_model_view,
                },
                {"text": "Back", "action": self._go_to_main_menu},
            ]

            self.menu_items[GameState.AI_TRAINING_MENU] = training_menu

        except Exception as e:
            print(f"Error refreshing training menu: {e}")

    def _refresh_training_setup_menu(self):
        """
        Refresh the training setup menu.
        Returns:
            None
        """
        try:
            from ai_training_manager import training_manager

            setup_menu = []

            difficulty_presets = training_manager.get_available_difficulty_presets()
            training_modes = training_manager.get_training_modes()

            setup_menu.append(
                {
                    "text": "=== AI DIFFICULTY SELECTION ===",
                    "action": None,
                    "enabled": False,
                }
            )

            for preset in difficulty_presets:
                preset_name = preset["name"]
                description = preset["description"]

                def create_difficulty_action(preset_data):
                    def action():
                        return self._select_training_difficulty(preset_data)

                    return action

                setup_menu.append(
                    {
                        "text": preset_name,
                        "action": create_difficulty_action(preset),
                        "enabled": True,
                        "description": description,
                    }
                )

            setup_menu.append({"text": "", "action": None, "enabled": False})
            setup_menu.append(
                {"text": "=== TRAINING MODE ===", "action": None, "enabled": False}
            )

            for mode in training_modes:
                mode_name = mode["name"]
                episodes = mode["episodes"]
                agents = mode["agents"]
                description = f"{episodes} episodes, {agents} agents"

                def create_mode_action(mode_data):
                    def action():
                        return self._select_training_mode(mode_data)

                    return action

                setup_menu.append(
                    {
                        "text": f"{mode_name}",
                        "action": create_mode_action(mode),
                        "enabled": True,
                        "description": description,
                    }
                )

            setup_menu.append({"text": "", "action": None, "enabled": False})

            can_start = (
                self.selected_training_difficulty is not None
                and self.selected_training_mode is not None
            )
            start_text = (
                "Start Training" if can_start else "Warning: Select difficulty and mode"
            )
            setup_menu.append(
                {
                    "text": start_text,
                    "action": self._start_ai_training if can_start else None,
                    "enabled": can_start,
                }
            )

            setup_menu.append(
                {"text": "Back", "action": self._go_to_ai_training, "enabled": True}
            )

            self.menu_items[GameState.TRAINING_SETUP_MENU] = setup_menu

        except Exception as e:
            print(f"Error refreshing training setup menu: {e}")

    def _select_training_difficulty(self, difficulty_preset):
        """
        Select the training difficulty.
        Returns:
            None
        """
        self.selected_training_difficulty = difficulty_preset
        preset_name = difficulty_preset.get("name", "Unknown")
        print(f"Selected training difficulty: {preset_name}")
        self._refresh_training_setup_menu()

    def _select_training_mode(self, training_mode):
        """
        Select the training mode.
        Returns:
            None
        """
        self.selected_training_mode = training_mode
        mode_name = training_mode.get("name", "Unknown")
        print(f"Selected training mode: {mode_name}")
        self._refresh_training_setup_menu()

    def _start_ai_training(self):
        """
        Start AI training.
        Returns:
            None
        """
        try:
            if not self.selected_training_difficulty or not self.selected_training_mode:
                print("Missing training parameters")
                return

            print("Starting AI training...")
            print(f"Difficulty: {self.selected_training_difficulty.get('name')}")
            print(f"Mode: {self.selected_training_mode.get('name')}")

            self._handle_state_change(GameState.TRAINING_GAME)

        except Exception as e:
            print(f"Error starting AI training: {e}")

    def _refresh_model_view_menu(self):
        """
        Refresh the model view menu.
        Returns:
            None
        """
        try:
            from ai_training_manager import training_manager

            models = training_manager.get_available_models()

            view_menu = []

            if not models:
                view_menu.append(
                    {
                        "text": "No trained models found",
                        "action": None,
                        "enabled": False,
                    }
                )
                view_menu.append(
                    {
                        "text": "Create models in training menu",
                        "action": None,
                        "enabled": False,
                    }
                )
            else:
                view_menu.append(
                    {
                        "text": f"=== FOUND MODELS: {len(models)} ===",
                        "action": None,
                        "enabled": False,
                    }
                )

                for model in models[:10]:
                    model_text = model["display_name"]
                    size_kb = model.get("size_kb", 0)
                    difficulty = model.get("difficulty", "Unknown")
                    training_type = model.get("training_type", "Unknown")

                    description = f"{difficulty} | {training_type} | {size_kb}KB"

                    view_menu.append(
                        {
                            "text": model_text,
                            "action": lambda m=model: self._select_model_for_game(m),
                            "enabled": True,
                            "description": description,
                        }
                    )

            view_menu.append({"text": "", "action": None, "enabled": False})
            view_menu.append(
                {"text": "Back", "action": self._go_to_ai_training, "enabled": True}
            )

            self.menu_items[GameState.MODEL_VIEW_MENU] = view_menu

        except Exception as e:
            print(f"Error refreshing model view menu: {e}")
            self.menu_items[GameState.MODEL_VIEW_MENU] = [
                {"text": "Error loading models", "action": None, "enabled": False},
                {"text": "Back", "action": self._go_to_ai_training, "enabled": True},
            ]

    def _select_model_for_game(self, model_info):
        """
        Select the model for game.
        Returns:
            None
        """
        try:
            self.selected_model_file = model_info["filename"]
            model_name = model_info.get("display_name", "Unknown")
            print(f"Selected model for game: {model_name}")
            self._handle_state_change(GameState.PLAYING_GAME)
        except Exception as e:
            print(f"Error selecting model: {e}")

    def _refresh_difficulty_menu(self):
        """
        Refresh the difficulty menu.
        Returns:
            None
        """
        difficulty_options = AIDifficultySelection.get_difficulty_options()

        difficulty_menu = []

        for option in difficulty_options:
            difficulty_menu.append(
                {
                    "text": option["name"],
                    "action": lambda opt=option: self._start_game_with_difficulty(
                        opt["preset_data"]
                    ),
                    "enabled": True,
                    "description": option["description"],
                }
            )

        difficulty_menu.append(
            {"text": "Back", "action": self._go_to_play_menu, "enabled": True}
        )

        self.menu_items[GameState.DIFFICULTY_MENU] = difficulty_menu

        if self.selected_index >= len(difficulty_menu):
            self.selected_index = 0

    def _start_game_with_model(self, model_info):
        """
        Start a game with a model.
        Returns:
            None
        """
        try:
            self.selected_model_file = model_info["filename"]
            print(
                f"Selected model: {model_info['display_name']} ({model_info['filename']})"
            )
            self._handle_state_change(GameState.PLAYING_GAME)
        except Exception as e:
            print(f"Error starting game with model {model_info['filename']}: {e}")
            self._go_to_play_menu()


if __name__ == "__main__":
    menu = MenuManager()
    menu.run()
