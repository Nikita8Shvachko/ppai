class AIDifficultyPresets:
    """
    Predefined AI difficulty settings that create human-like behavior
    instead of godlike perfect play.
    """

    BEGINNER = {
        "name": "Beginner",
        "prediction_time_limit": 0.3,
        "distance_error_factor": 60,
        "base_prediction_error": 20,
        "wall_bounce_accuracy": 0.7,
        "description": "Very human-like, makes lots of mistakes",
    }

    CASUAL = {
        "name": "Casual Player",
        "prediction_time_limit": 0.4,
        "distance_error_factor": 45,
        "base_prediction_error": 15,
        "wall_bounce_accuracy": 0.8,
        "description": "Like a casual human player",
    }

    SKILLED = {
        "name": "Skilled Player",
        "prediction_time_limit": 0.5,
        "distance_error_factor": 30,
        "base_prediction_error": 10,
        "wall_bounce_accuracy": 0.9,
        "description": "Like an experienced human player",
    }

    EXPERT = {
        "name": "Expert Player",
        "prediction_time_limit": 0.6,
        "distance_error_factor": 20,
        "base_prediction_error": 8,
        "wall_bounce_accuracy": 0.95,
        "description": "Like a human expert, but still makes small errors",
    }

    UNBEATABLE = {
        "name": "Unbeatable (Original)",
        "prediction_time_limit": 10.0,
        "distance_error_factor": 0,
        "base_prediction_error": 0,
        "wall_bounce_accuracy": 1.0,
        "description": "Original godlike AI (not fun to play against)",
    }

    @classmethod
    def get_all_presets(cls):
        """Get all available difficulty presets
        Args:
            cls (class): The class to get the presets from
        Returns:
            list: A list of difficulty presets
        """
        return [cls.BEGINNER, cls.CASUAL, cls.SKILLED, cls.EXPERT, cls.UNBEATABLE]

    @classmethod
    def get_preset(cls, name):
        """Get a specific preset by name
        Args:
            name (str): The name of the preset to get
        Returns:
            dict: The preset dictionary
        """
        presets = {
            "beginner": cls.BEGINNER,
            "casual": cls.CASUAL,
            "skilled": cls.SKILLED,
            "expert": cls.EXPERT,
            "unbeatable": cls.UNBEATABLE,
        }
        return presets.get(name.lower(), cls.CASUAL)
