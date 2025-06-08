import os

import pygame

from menu_manager import GameState, MenuManager
from parallel_ai_controller import ParallelAIController
from pong_game import Game


class GameLauncher:
    """Main class for launching the game with menu system"""

    def __init__(self):
        """
        Initialize the GameLauncher.
        Returns:
            None
        """
        try:
            pygame.init()

            self.menu_manager = MenuManager()
            self.current_game = None

            self.current_state = GameState.MAIN_MENU
            self.selected_index = 0
            self.width = 1200
            self.height = 900
            self.menu_items = {}
            self.game_just_exited = False

            self.game_settings = {
                "difficulty": None,
                "two_players": False,
                "fast_training": False,
                "reset_model": False,
                "use_best_model": False,
            }
            self._resources_initialized = True

            self._integrate_with_menu()

        except Exception as e:
            print(f"Error initializing game launcher: {e}")
            self._resources_initialized = False
            self.cleanup()
            raise

    def cleanup(self):
        """
        Cleanup launcher resources.
        Returns:
            None
        """
        try:
            if hasattr(self, "current_game") and self.current_game:
                try:
                    self.current_game.cleanup()
                except Exception as e:
                    print(f"Error cleaning up game: {e}")
                self.current_game = None

            if hasattr(self, "menu_manager") and self.menu_manager:
                try:
                    self.menu_manager.cleanup()
                except Exception as e:
                    print(f"Error cleaning up menu: {e}")

            self._resources_initialized = False
            self.game_settings = {
                "difficulty": None,
                "two_players": False,
                "fast_training": False,
                "reset_model": False,
                "use_best_model": False,
            }

        except Exception as e:
            print(f"Error during launcher cleanup: {e}")
        finally:
            try:
                if pygame.get_init():
                    pygame.quit()
            except Exception:
                pass

    def __del__(self):
        """
        Destructor for guaranteed resource cleanup.
        Returns:
            None
        """
        if self._resources_initialized:
            self.cleanup()

    def _validate_settings(self):
        """
        Validate game settings for compatibility.
        Returns:
            bool: True if settings are valid, False otherwise
        """
        try:
            if self.game_settings["two_players"]:
                if self.game_settings["difficulty"] is not None:
                    print("Warning: Difficulty ignored in two player mode")
                if self.game_settings["use_best_model"]:
                    print("Warning: Best model ignored in two player mode")

            if self.game_settings["use_best_model"]:
                if self.game_settings["difficulty"] is not None:
                    print("Warning: Difficulty ignored when using best model")

            return True

        except Exception as e:
            print(f"Error validating settings: {e}")
            return False

    def _integrate_with_menu(self):
        """
        Integrate launcher methods with menu manager.
        Returns:
            None
        """

        def start_pong_game_wrapper(*args, **kwargs):
            model_file = getattr(self.menu_manager, "selected_model_file", None)
            return self._start_pong_game(*args, model_file=model_file, **kwargs)

        self.menu_manager._start_pong_game = start_pong_game_wrapper
        self.menu_manager._start_pong_training = self._start_pong_training
        self.menu_manager._start_ai_vs_ai_mode = self._start_ai_vs_ai_mode
        self.menu_manager._run_training_mode = self._run_training_mode
        self.menu_manager._run_playing_mode = self._run_playing_mode

        self.current_state = self.menu_manager.current_state
        self.selected_index = self.menu_manager.selected_index
        self.menu_items = self.menu_manager.menu_items

    def run(self):
        """
        Main application loop.
        Returns:
            None
        """
        if not self._resources_initialized:
            print("Launcher resources not properly initialized")
            return

        try:
            while self.menu_manager.running:
                self.current_state = self.menu_manager.current_state
                self.selected_index = self.menu_manager.selected_index

                self.menu_manager.handle_events()

                if self.game_just_exited:
                    self.game_just_exited = False
                    continue

                if self.menu_manager.current_state == GameState.TRAINING_GAME:
                    self._run_training_mode()
                elif self.menu_manager.current_state == GameState.PLAYING_GAME:
                    self._run_playing_mode()
                elif self.menu_manager.current_state == GameState.ERROR_STATE:
                    print("Error state detected, returning to main menu")
                    self.menu_manager._handle_state_change(GameState.MAIN_MENU)
                else:
                    self.menu_manager.draw()
                    if hasattr(self.menu_manager, "clock"):
                        self.menu_manager.clock.tick(60)

        except pygame.error as e:
            print(f"Pygame error in launcher loop: {e}")
        except Exception as e:
            print(f"Error in launcher loop: {e}")
        finally:
            self.cleanup()

    def _start_pong_game(
        self,
        difficulty=None,
        use_best_model=False,
        two_players=False,
        model_file=None,
    ):
        """
        Configure parameters for game launch.
        Returns:
            None
        """
        try:
            self.game_settings["difficulty"] = difficulty
            self.game_settings["two_players"] = two_players
            self.game_settings["use_best_model"] = use_best_model
            self.game_settings["selected_model_file"] = model_file

            if not self._validate_settings():
                print("Invalid game settings, using defaults")
                self.game_settings = {
                    "difficulty": None,
                    "two_players": False,
                    "fast_training": False,
                    "reset_model": False,
                    "use_best_model": False,
                    "selected_model_file": None,
                }

            print(f"Game settings: {self.game_settings}")

        except Exception as e:
            print(f"Error setting up game: {e}")
            self.game_settings = {
                "difficulty": None,
                "two_players": False,
                "fast_training": False,
                "reset_model": False,
                "use_best_model": False,
                "selected_model_file": None,
            }

    def _start_pong_training(self, reset_model=False, fast_mode=False):
        """
        Configure parameters for training.
        Returns:
            None
        """
        try:
            self.game_settings["reset_model"] = reset_model
            self.game_settings["fast_training"] = fast_mode
            self.game_settings["agent_vs_agent"] = True
            self.game_settings["training_mode"] = True
            print(f"Training settings: {self.game_settings}")
        except Exception as e:
            print(f"Error setting up training: {e}")
            self.game_settings["reset_model"] = False
            self.game_settings["fast_training"] = False
            self.game_settings["agent_vs_agent"] = False
            self.game_settings["training_mode"] = False

    def _start_ai_vs_ai_mode(self):
        """
        Start AI vs AI visual mode (for watching two AIs play).
        Returns:
            None
        """
        try:
            self.game_settings = {
                "difficulty": None,
                "two_players": False,
                "fast_training": False,
                "reset_model": False,
                "use_best_model": False,
                "agent_vs_agent": True,
                "training_mode": True,
                "visual_mode": True,
            }
            print(f"AI vs AI mode settings: {self.game_settings}")
        except Exception as e:
            print(f"Error setting up AI vs AI mode: {e}")
            self.game_settings = {
                "difficulty": None,
                "two_players": False,
                "fast_training": False,
                "reset_model": False,
                "use_best_model": False,
                "agent_vs_agent": False,
                "training_mode": False,
                "visual_mode": False,
            }

    def _run_training_mode(self):
        """
        Run training mode or visual AI vs AI mode.
        Returns:
            None
        """

        if self.game_settings.get("visual_mode", False):
            print("Starting visual AI vs AI mode...")
            self._run_visual_ai_vs_ai()
            return

        print("Starting deep AI training mode...")

        difficulty_preset = getattr(
            self.menu_manager, "selected_training_difficulty", None
        )
        training_mode = getattr(self.menu_manager, "selected_training_mode", None)

        if not difficulty_preset or not training_mode:
            print("Missing training parameters. Using defaults.")
            from ai_difficulty_presets import AIDifficultyPresets
            from ai_training_manager import TrainingMode

            difficulty_preset = AIDifficultyPresets.get_all_presets()[0]
            training_mode = TrainingMode.STANDARD.value

        if not self.current_game:
            self.current_game = Game(training_mode=True)

        try:
            from ai_training_manager import training_manager

            mock_paddle = self.current_game.opponent
            mock_ball = self.current_game.ball

            print(
                f"Starting deep training with {training_mode.get('name', 'Unknown')} mode"
            )
            print(f"AI Difficulty: {difficulty_preset.get('name', 'Unknown')}")
            print(f"Episodes: {training_mode.get('episodes', 'Unknown')}")
            print(f"Agents: {training_mode.get('agents', 'Unknown')}")

            success = training_manager.start_training(
                difficulty_preset=difficulty_preset,
                training_mode=training_mode,
                paddle=mock_paddle,
                ball=mock_ball,
                sync=True,
            )

            if success:
                print("Training completed successfully!")
                print("Trained model saved to ai_models directory")

                self.menu_manager.selected_training_difficulty = None
                self.menu_manager.selected_training_mode = None
            else:
                print("Training failed")

        except ImportError as e:
            print(f"Training system not available: {e}")
            print("Falling back to basic training mode...")
            self._run_basic_training_fallback()
        except Exception as e:
            print(f"Training error: {e}")
            print("Falling back to basic training mode...")
            self._run_basic_training_fallback()
        finally:
            if self.current_game:
                self._cleanup_game()
            self.game_just_exited = True
            self.menu_manager._handle_state_change(GameState.AI_TRAINING_MENU)

    def _run_visual_ai_vs_ai(self):
        """
        Run visual mode where two AIs play against each other.
        Returns:
            None
        """
        print("Starting visual AI vs AI mode...")

        if not self.current_game:
            self.current_game = Game(training_mode=True)

            self.current_game.player.is_ai = True
            self.current_game.opponent.is_ai = True

            self.current_game.fast_training = False
            self.current_game.simulation_speed = 1.0

            print("Both paddles are now AI-controlled")
            print("Use 'T' key to toggle training speed")
            print("Use 'V' key to show neural networks")
            print("Use 'M' key to show metrics")

        try:
            self.current_game.run()
        except KeyboardInterrupt:
            print("Visual AI vs AI mode interrupted by user")
        finally:
            if self.current_game:
                self._cleanup_game()
            self.game_just_exited = True
            self.menu_manager._handle_state_change(GameState.AI_TRAINING_MENU)

    def _run_basic_training_fallback(self):
        """
        Fallback to basic training if new system fails.
        Returns:
            None
        """
        try:
            if not self.current_game.opponent.ai_controller:
                self.current_game.opponent.ai_controller = ParallelAIController(
                    self.current_game.opponent,
                    self.current_game.ball,
                    num_agents=4,
                    input_size=12,
                    hidden_size=32,
                    output_size=1,
                    exchange_interval=1000,
                    mutation_rate=0.1,
                    batch_size=32,
                )
                self.current_game.opponent.ai_controller.start_training()
                print("Basic AI training initialized")

            print("Running basic training session...")
            self.current_game.run()

        except Exception as e:
            print(f"Fallback training also failed: {e}")

    def _run_playing_mode(self):
        """
        Run game mode.
        Returns:
            None
        """
        print("Starting game mode...")

        if not self.current_game:
            self.current_game = Game(training_mode=False)

            is_two_player = getattr(self.menu_manager, "two_player_mode", False)
            if is_two_player:
                self.game_settings["two_players"] = True

            if self.game_settings.get("two_players") or is_two_player:
                self._setup_two_players()
                print("Two-player mode: Skipping AI initialization")
            else:
                selected_model = getattr(self.menu_manager, "selected_model_file", None)

                if selected_model:
                    print(f"Loading selected model: {selected_model}")
                    self._load_model_for_game(selected_model)
                elif self.game_settings.get("use_best_model"):
                    print("Loading best available model...")
                    self._initialize_default_ai()
                    self._load_best_model()
                else:
                    print("Setting up difficulty-based AI...")
                    self._initialize_default_ai()

                difficulty = getattr(self.menu_manager, "selected_difficulty", None)
                if difficulty:
                    self._setup_difficulty(difficulty)
                    print(f"Applied difficulty: {difficulty.get('name', 'Unknown')}")

        try:
            self.current_game.run()
        except KeyboardInterrupt:
            print("Game interrupted by user")
        finally:
            if self.current_game:
                self._cleanup_game()
            self.game_just_exited = True
            self.menu_manager._handle_state_change(GameState.PLAY_MENU)

    def _initialize_default_ai(self):
        """
        Initialize default AI controller for non-difficulty models.
        Returns:
            None
        """
        if not self.current_game:
            return

        if self.game_settings.get("agent_vs_agent", False):
            if not self.current_game.player.ai_controller:
                self.current_game.player.ai_controller = ParallelAIController(
                    self.current_game.player,
                    self.current_game.ball,
                    num_agents=2,
                    input_size=12,
                    hidden_size=32,
                    output_size=1,
                    exchange_interval=2000,
                    mutation_rate=0.05,
                    batch_size=16,
                    difficulty_preset=self.game_settings.get("difficulty"),
                )
                self.current_game.player.ai_controller.start_training()
                print("Player AI controller initialized")

            if not self.current_game.opponent.ai_controller:
                self.current_game.opponent.ai_controller = ParallelAIController(
                    self.current_game.opponent,
                    self.current_game.ball,
                    num_agents=2,
                    input_size=12,
                    hidden_size=32,
                    output_size=1,
                    exchange_interval=2000,
                    mutation_rate=0.05,
                    batch_size=16,
                    difficulty_preset=self.game_settings.get("difficulty"),
                )
                self.current_game.opponent.ai_controller.start_training()
                print("Opponent AI controller initialized")
        else:
            if not self.current_game.opponent.ai_controller:
                self.current_game.opponent.ai_controller = ParallelAIController(
                    self.current_game.opponent,
                    self.current_game.ball,
                    num_agents=2,
                    input_size=12,
                    hidden_size=32,
                    output_size=1,
                    exchange_interval=2000,
                    mutation_rate=0.05,
                    batch_size=16,
                    difficulty_preset=self.game_settings.get("difficulty"),
                )
                self.current_game.opponent.ai_controller.start_training()
                print("Opponent AI controller initialized")

    def _setup_difficulty(self, difficulty_preset):
        """
        Setup AI difficulty using new human-like presets.
        Returns:
            None
        """
        try:
            if not self.current_game or not difficulty_preset:
                return

            opponent = self.current_game.opponent
            if hasattr(opponent, "ai_controller") and opponent.ai_controller:
                opponent.ai_controller.apply_difficulty_preset(difficulty_preset)
                preset_name = difficulty_preset.get("name", "Unknown")
                print(f"Applied AI difficulty preset: {preset_name}")
            else:
                print("No AI controller found to apply difficulty to")

        except Exception as e:
            print(f"Error setting difficulty: {e}")

    def _setup_two_players(self):
        """
        Setup mode for two players.
        Returns:
            None
        """
        try:
            if self.current_game:
                self.current_game.set_two_player_mode(True)
                print("Two player mode enabled")
        except Exception as e:
            print(f"Error setting up two player mode: {e}")

    def _load_model_for_game(self, model_filename: str):
        """
        Load specified model for AI in current game.
        Returns:
            None
        """
        try:
            if not self.current_game:
                print(f"Cannot load model {model_filename}: No game instance")
                return

            models_dir = os.path.join(os.path.dirname(__file__), "ai_models")
            model_path = os.path.join(models_dir, model_filename)

            if os.path.exists(model_path):
                if self.current_game.opponent.ai_controller:
                    self.current_game.opponent.ai_controller.load_model(model_path)
                    print(f"Loaded model from {model_filename}")
                else:
                    print("No AI controller to load model into")
            else:
                print(f"Model file {model_filename} not found")

        except Exception as e:
            print(f"Error loading model {model_filename}: {e}")
            try:
                if not self.current_game.opponent.ai_controller:
                    self.current_game.opponent.ai_controller = ParallelAIController(
                        self.current_game.opponent,
                        self.current_game.ball,
                        num_agents=2,
                        input_size=12,
                        hidden_size=32,
                        output_size=1,
                        exchange_interval=2000,
                        mutation_rate=0.05,
                        batch_size=16,
                        difficulty_preset=self.game_settings.get("difficulty"),
                    )
                    print("Created fallback AI controller")
            except Exception as fallback_error:
                print(f"Fallback failed: {fallback_error}")

    def _load_best_model(self):
        """
        Load best trained model.
        Returns:
            None
        """
        print("Attempting to load best_model.pkl...")
        self._load_model_for_game("best_model.pkl")

    def _cleanup_game(self):
        """
        Cleanup game resources.
        Returns:
            None
        """
        if self.current_game:
            if (
                self.current_game.training_mode
                and hasattr(self.current_game.opponent, "ai_controller")
                and self.current_game.opponent.ai_controller
            ):
                ai_ctrl = self.current_game.opponent.ai_controller
                if hasattr(ai_ctrl, "save_model"):
                    ai_ctrl.save_model("ai_models/current_model.pkl")
                    print("AI model saved")

            self.current_game.cleanup()
            self.current_game = None

    def _handle_state_change(self, new_state: GameState):
        """
        Safe menu state change (delegates to menu_manager).
        Returns:
            None
        """
        try:
            if hasattr(self.menu_manager, "_handle_state_change"):
                self.menu_manager._handle_state_change(new_state)
                self.current_state = self.menu_manager.current_state
                self.selected_index = self.menu_manager.selected_index
            else:
                print("Menu manager does not support state changes")

        except Exception as e:
            print(f"Error changing state: {e}")
            if hasattr(self.menu_manager, "_handle_state_change"):
                self.menu_manager._handle_state_change(GameState.ERROR_STATE)

    def _is_valid_state_transition(self, current: GameState, new: GameState) -> bool:
        """
        Check validity of state transition (delegates to menu_manager).
        Returns:
            bool: True if the state transition is valid, False otherwise
        """
        if hasattr(self.menu_manager, "_is_valid_state_transition"):
            return self.menu_manager._is_valid_state_transition(current, new)
        return True

    def _get_menu_item_rect(self, index: int) -> pygame.Rect:
        """
        Get menu item rectangle.
        Returns:
            pygame.Rect: The menu item rectangle
        """
        current_menu = self.menu_items.get(self.current_state, [])
        if not current_menu:
            return pygame.Rect(0, 0, 0, 0)

        start_y = self.height // 2 - (len(current_menu) * 50) // 2
        item_y = start_y + index * 60

        item_width = 400
        item_height = 40
        item_x = self.width // 2 - item_width // 2

        return pygame.Rect(item_x, item_y - item_height // 2, item_width, item_height)

    def _execute_selected_action(self):
        """
        Execute the currently selected menu action.
        Returns:
            None
        """
        try:
            if hasattr(self.menu_manager, "_execute_selected_action"):
                self.menu_manager._execute_selected_action()
            else:
                print("No action execution method available")
        except Exception as e:
            print(f"Error executing selected action: {e}")

    def _handle_mouse_click(self, pos):
        """
        Handle mouse click.
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
        Handle mouse hover.
        Returns:
            None
        """
        current_menu = self.menu_items.get(self.current_state, [])
        for i, item in enumerate(current_menu):
            item_rect = self._get_menu_item_rect(i)
            if item_rect.collidepoint(pos):
                self.selected_index = i
                break


if __name__ == "__main__":
    launcher = GameLauncher()
    launcher.run()
