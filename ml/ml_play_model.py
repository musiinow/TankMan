import random
from typing import Optional
import pygame
import os
import numpy as np
from stable_baselines3 import PPO
from mlgame.utils.enum import get_ai_name
import math
from src.env import BACKWARD_CMD, FORWARD_CMD, TURN_LEFT_CMD, TURN_RIGHT_CMD, SHOOT, AIM_LEFT_CMD, AIM_RIGHT_CMD
import random

WIDTH = 1000 # pixel
HEIGHT = 600 # pixel
TANK_SPEED = 8 # pixel
CELL_PIXEL_SIZE = 50 # pixel
DEGREES_PER_SEGMENT = 45 # degree

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # 上層資料夾
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_AIM_PATH = os.path.join("aim.zip")
MODEL_CHASE_PATH = os.path.join("chase.zip")

COMMAND_AIM = [
    ["NONE"],
    [AIM_LEFT_CMD],
    [AIM_RIGHT_CMD],
    [SHOOT],   
]

COMMAND_CHASE = [
    ["NONE"],
    [FORWARD_CMD],
    [BACKWARD_CMD],
    [TURN_LEFT_CMD],
    [TURN_RIGHT_CMD],
]

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        """
        Constructor

        @param side A string like "1P" or "2P" indicates which player the `MLPlay` is for.
        """
        self.side = ai_name
        print(f"Initial Game {ai_name} ML script")
        self.time = 0
        self.player: str = "1P"

        # Load the trained models
        self.model_aim = PPO.load(MODEL_AIM_PATH)
        self.model_chase = PPO.load(MODEL_CHASE_PATH)

        self.target_x = None
        self.target_y = None


    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        self._scene_info = scene_info
        enemies = scene_info.get("competitor_info", [])

        player_x = scene_info["x"]
        player_y = scene_info["y"]
        cooldown = scene_info.get("cooldown", 0)

        nearest_enemy = min(enemies, key=lambda e: math.dist((player_x, player_y), (e["x"], e["y"])))
        enemy_x, enemy_y = nearest_enemy["x"], nearest_enemy["y"]

        distance = math.dist((player_x, player_y), (enemy_x, enemy_y))

        print(f"Nearest enemy distance: {distance}, Cooldown: {cooldown}")

        self.target_x = enemy_x
        self.target_y = enemy_y

        if self.target_x is None or self.target_y is None:
            print("No valid target available.")
            return "RESET"
        enemy_player_x = abs(enemy_x - player_x)
        enemy_player_y = abs(enemy_y - player_y)
        # Randomly switch between model_aim and model_chase
        if distance <= 300 and cooldown == 0 and (enemy_x == player_x or enemy_y == player_y or enemy_player_x == enemy_player_y):
            obs = self._get_obs_aim()
            action, _ = self.model_aim.predict(obs, deterministic=True)
            command = COMMAND_AIM[action]
        else:
            obs = self._get_obs_chase()
            action, _ = self.model_chase.predict(obs, deterministic=True)
            command = COMMAND_CHASE[action]

        print(f"Target is : ({self.target_x, self.target_y})")
        print(f"Predicted action: {command}")
        self.time += 1
        return command


    def reset(self):
        """
        Reset the status
        """
        print(f"Resetting Game {self.side}")

    def get_obs_chase(self, player: str, target_x: int, target_y: int, scene_info: dict) -> np.ndarray:
        player_x = scene_info.get("x", 0)
        player_y = scene_info.get("y", 0)
        tank_angle = scene_info.get("angle", 0) + 180
        tank_angle_index: int = self._angle_to_index(tank_angle)
        dx = target_x - player_x
        dy = target_y - player_y
        angle_to_target = math.degrees(math.atan2(dy, dx))
        angle_to_target_index: int = self._angle_to_index(angle_to_target)
        obs = np.array([float(tank_angle_index), float(angle_to_target_index)], dtype=np.float32)
        print("Chase obs: " + str(obs))
        return obs

    def get_obs_aim(self, player: str, target_x: int, target_y: int, scene_info: dict) -> np.ndarray:
        player_x = scene_info.get("x", 0)
        player_y = scene_info.get("y", 0)
        gun_angle = scene_info.get("gun_angle", 0) + scene_info.get("angle", 0) + 180
        gun_angle_index: int = self._angle_to_index(gun_angle)
        dx = target_x - player_x
        dy = target_y - player_y 
        print(f'player_x{player_x},player_y{player_y}')
        print(f'target_x{target_x},traget_y{target_y}')
        print(f'dx{dx},dy{dy}')
        angle_to_target = math.degrees(math.atan2(dy, dx))
        angle_to_target_index: int = self._angle_to_index(angle_to_target)
        print("Aim angle: " + str(angle_to_target))
        obs = np.array([float(gun_angle_index), float(angle_to_target_index)], dtype=np.float32)
        return obs

    def _get_obs_chase(self) -> np.ndarray:
        return self.get_obs_chase(
            self.player,
            self.target_x,
            self.target_y,
            self._scene_info,
        )

    def _get_obs_aim(self) -> np.ndarray:
        return self.get_obs_aim(
            self.player,
            self.target_x,
            self.target_y,
            self._scene_info,
        )

    def _angle_to_index(self, angle: float) -> int:
        angle = (angle + 360) % 360

        segment_center = (angle + DEGREES_PER_SEGMENT/2) // DEGREES_PER_SEGMENT
        return int(segment_center % (360 // DEGREES_PER_SEGMENT))