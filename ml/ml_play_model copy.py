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
MODEL_AIM_PATH = os.path.join(MODEL_DIR, "aim.zip")##############
MODEL_CHASE_PATH = os.path.join(MODEL_DIR, "chase.zip")#############

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
        #print("scene_info keys:", scene_info)
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        
        #self._scene_info = scene_info

        enemies = scene_info.get("competitor_info", [])

        player_x = scene_info["x"]
        player_y = scene_info["y"]
        cooldown = scene_info.get("cooldown", 0)

        nearest_enemy = min(enemies, key=lambda e: math.dist((player_x, player_y), (e["x"], e["y"])))
        enemy_x, enemy_y = nearest_enemy["x"], nearest_enemy["y"]

        distance = math.dist((player_x, player_y), (enemy_x, enemy_y))

        print(f"Nearest enemy distance: {distance}, Cooldown: {cooldown}")

        enemy_player_x = abs(enemy_x - player_x)
        enemy_player_y = abs(enemy_y - player_y)
        if distance <= 300 and cooldown == 0 and (enemy_x == player_x or enemy_y == player_y or enemy_player_x == enemy_player_y):
            obs = self.get_obs_aim(self.player, enemy_x, enemy_y, scene_info)
            action, _ = self.model_aim.predict(obs, deterministic=True)
            command = COMMAND_AIM[action]
        else:
            obs = self.get_obs_chase(self.player, scene_info)
            action, _ = self.model_chase.predict(obs, deterministic=True)
            command = COMMAND_CHASE[action]

        print(f"Predicted action: {command}")
        self.time += 1
        return command



    def reset(self):
        """
        Reset the status
        """
        print(f"Resetting Game {self.side}")

    def get_obs_chase(self, player, scene_info) -> np.ndarray:
        player_data = scene_info
        x, y = player_data["x"], player_data["y"]
        tank_angle = player_data["angle"] 
        oil = player_data["oil"] 
        bullets = player_data["power"] 


        def get_8_direction(angle):
            return round(angle / 45) % 8

        tank_angle_index = get_8_direction(tank_angle)


        nearest_enemy = None
        min_enemy_dist = float("inf")
        angle_to_enemy_index = 0
        for enemy in player_data["competitor_info"]:
            enemy_x, enemy_y = enemy["x"], enemy["y"]
            dist = math.sqrt((enemy_x - x) ** 2 + (enemy_y - y) ** 2)
            if dist < min_enemy_dist:
                min_enemy_dist = dist
                nearest_enemy = enemy

        if nearest_enemy:
            angle_to_enemy = math.degrees(math.atan2(nearest_enemy["y"] - y, nearest_enemy["x"] - x))
            angle_to_enemy_index = get_8_direction(angle_to_enemy)


        nearest_oil = None
        min_oil_dist = float("inf")
        angle_to_oil_index = 0
        for oil_station in player_data["oil_stations_info"]:
            oil_x, oil_y = oil_station["x"], oil_station["y"]
            dist = math.sqrt((oil_x - x) ** 2 + (oil_y - y) ** 2)
            if dist < min_oil_dist:
                min_oil_dist = dist
                nearest_oil = oil_station

        if nearest_oil:
            angle_to_oil = math.degrees(math.atan2(nearest_oil["y"] - y, nearest_oil["x"] - x))
            angle_to_oil_index = get_8_direction(angle_to_oil)


        nearest_bullet = None
        min_bullet_dist = float("inf")
        angle_to_bullet_index = 0
        for bullet_station in player_data["bullet_stations_info"]:
            bullet_x, bullet_y = bullet_station["x"], bullet_station["y"]
            dist = math.sqrt((bullet_x - x) ** 2 + (bullet_y - y) ** 2)
            if dist < min_bullet_dist:
                min_bullet_dist = dist
                nearest_bullet = bullet_station

        if nearest_bullet:
            angle_to_bullet = math.degrees(math.atan2(nearest_bullet["y"] - y, nearest_bullet["x"] - x))
            angle_to_bullet_index = get_8_direction(angle_to_bullet)


        normalized_enemy_dist = min(min_enemy_dist / 100, 10) 
        normalized_oil_dist = min(min_oil_dist / 100, 10)
        normalized_bullet_dist = min(min_bullet_dist / 100, 10)


        obs = np.array([
            tank_angle_index,
            angle_to_enemy_index,
            normalized_enemy_dist,
            angle_to_oil_index,
            normalized_oil_dist,
            angle_to_bullet_index,
            normalized_bullet_dist,
            oil,
            bullets
        ], dtype=np.float32)
        print("Chase obs: " + str(obs))
        return obs

    def get_obs_aim(self, player: str, target_x: int, target_y: int, scene_info: dict) -> np.ndarray:
        player_x = scene_info.get("x", 0)
        player_y = -scene_info.get("y", 0)
        gun_angle = scene_info.get("gun_angle", 0) + scene_info.get("angle", 0) + 180
        gun_angle_index: int = self._angle_to_index(gun_angle)
        dx = target_x - player_x
        dy = target_y - player_y 
        angle_to_target = math.degrees(math.atan2(dy, dx))
        angle_to_target_index: int = self._angle_to_index(angle_to_target)
        print("Aim angle: " + str(angle_to_target))
        obs = np.array([float(gun_angle_index), float(angle_to_target_index)], dtype=np.float32)
        return obs

    def _get_obs_chase(self) -> np.ndarray:
        return self.get_obs_chase(
            self.player,
            self._scene_info
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
