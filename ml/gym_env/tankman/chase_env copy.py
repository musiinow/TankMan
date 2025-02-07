import sys
from os import path
sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
)

import math
import random
from typing import Optional

import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from mlgame.utils.enum import get_ai_name

from src.env import FORWARD_CMD, BACKWARD_CMD, TURN_LEFT_CMD, TURN_RIGHT_CMD
from .base_env import TankManBaseEnv

WIDTH = 1000 # pixel
HEIGHT = 600 # pixel
TANK_SPEED = 8 # pixel
CELL_PIXEL_SIZE = 50 # pixel
DEGREES_PER_SEGMENT = 45 # degree
COMMAND = [
    ["NONE"],
    [FORWARD_CMD],
    [BACKWARD_CMD],
    [TURN_LEFT_CMD],
    [TURN_RIGHT_CMD],
]

class ResupplyEnv(TankManBaseEnv):
    def __init__(
        self,
        green_team_num: int,
        blue_team_num: int,
        frame_limit: int,
        player: Optional[str] = None,
        supply_type: Optional[str] = None,
        randomize: Optional[bool] = False,
        sound: str = "off",
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__(green_team_num, blue_team_num, frame_limit, sound, render_mode)

        self.player_num = green_team_num + blue_team_num
        self.randomize = randomize

        if self.randomize:
            self.player = get_ai_name(np.random.randint(self.player_num))
            self.supply_type = np.random.choice(["oil_stations", "bullet_stations"])
        else:
            assert player is not None and supply_type is not None
            assert player in [
                get_ai_name(i) for i in range(self.player_num)
            ], f"{player} is not a valid player id"
            assert supply_type in [
                "oil_stations",
                "bullet_stations",
            ], f"{supply_type} is not a valid supply type"

            self.player = player
            self.supply_type = supply_type

        self._total_angle_segment: int = 360 // DEGREES_PER_SEGMENT
        if type(self._total_angle_segment) is not int:
            raise ValueError("The total angle segment should be an integer, please modify the DEGREES_PER_SEGMENT value.")
        
        # Initialize target position
        #self.target_x = random.randint(CELL_PIXEL_SIZE, WIDTH - 2 * CELL_PIXEL_SIZE)
        #self.target_y = random.randint(CELL_PIXEL_SIZE, HEIGHT - 2 * CELL_PIXEL_SIZE)

        # gun_angle, angle_to_target
        self._observation_space = Box(low=0, high=10, shape=(9,), dtype=np.float32)

        self._action_space = Discrete(len(COMMAND))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        if self.randomize:
            self.player = get_ai_name(np.random.randint(self.player_num))

        return super().reset(seed=seed, options=options)

    @property
    def observation_space(self) -> Box:
        return self._observation_space

    @property
    def action_space(self) -> Discrete:
        return self._action_space

    def get_obs(self, player, scene_info) -> np.ndarray:
        player_data = scene_info[self.player]
        x, y = player_data["x"], player_data["y"]
        tank_angle = player_data["angle"]  # 自己的方向（0~360 度）
        oil = player_data["oil"]  # 燃油
        oil = min(oil / 10, 10)
        bullets = player_data["power"]  # 子彈數量


        def get_8_direction(angle):
            return round(angle / 45) % 8  # 360 度 / 8 方向 = 每個方向 45 度

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

        # 正規化距離（以 100 為單位）
        normalized_enemy_dist = min(min_enemy_dist / 100, 10)  # 最大值限制為 10
        normalized_oil_dist = min(min_oil_dist / 100, 10)
        normalized_bullet_dist = min(min_bullet_dist / 100, 10)

        tank_angle_index = math.floor(tank_angle_index)
        angle_to_enemy_index = math.floor(angle_to_enemy_index)


        # 組合 obs 陣列
        obs = np.array([
            tank_angle_index,      # 自己的方向 (0~7)
            angle_to_enemy_index,  # 相對敵人的方向 (0~7)
            normalized_enemy_dist, # 與敵人的距離 (0~10)
            angle_to_oil_index,    # 相對燃油站的方向 (0~7)
            normalized_oil_dist,   # 與燃油站的距離 (0~10)
            angle_to_bullet_index, # 相對子彈補給站的方向 (0~7)
            normalized_bullet_dist,# 與子彈補給站的距離 (0~10)
            oil,                   # 燃油 (0~100)
            bullets                # 子彈數量 (0~10)
        ], dtype=np.float32)

        return obs


    def get_reward(self, obs: np.ndarray, action: int) -> float:
        obs[0] = int(obs[0])
        obs[1] = int(obs[1])
        tank_angle_index = obs[0]
        angle_to_enemy_index = obs[1]
        distance_to_enemy = obs[2] * 100
        angle_to_oil_index = obs[3]
        distance_to_oil = obs[4] * 100
        angle_to_bullet_index = obs[5]
        distance_to_bullet = obs[6] * 100
        oil = obs[7] * 10
        bullets = obs[8] 

        reward = 0.0

        # the angle is point at the right side of the target
        if obs[0] in [(obs[1] + x) % 8 for x in [5, 6, 7]] and action == 3: # TURN_LEFT_CMD
            reward += -2
        elif obs[0] in [(obs[1] + x) % 8 for x in [5, 6, 7]] and action == 4:   # TURN_RIGHT_CMD
            reward += 5

        # the angle is point at the left side of the target
        elif obs[0] in [(obs[1] + x) % 8 for x in [1, 2, 3]] and action == 4:   # TURN_RIGHT_CMD
            reward += 5
        elif obs[0] in [(obs[1] + x) % 8 for x in [1, 2, 3]] and action == 3:   # TURN_LEFT_CMD
            reward += -2

        # the angel is point on enemy
        elif obs[0] == obs[1] and action == 1 and distance_to_enemy >= 300:
            reward += 8
        elif obs[0] == obs[1] and action == 2 and distance_to_enemy >= 300:
            reward += -4
        elif obs[0] == obs[1] and action == 1 and distance_to_enemy < 200:
            reward += -1
        elif obs[0] == obs[1] and action == 2 and distance_to_enemy < 200:
            reward += 2

        elif obs[0] == obs[1] and (action !=2 or action != 1):
            reward += -20    

        elif action == 0:
            reward -= 3

        #print("obs 0 and 1",type(obs[0]), type(obs[1]))
            

        # if oil < 100:
        #     reward -= 1.0 
        #     if distance_to_oil > 0:
        #         if angle_to_oil_index == tank_angle_index:  # 朝向補給站
        #             reward += 3.0
        #         else:
        #             reward -= 1.0
        #     if action == 1:
        #         reward += 2.0


        # if bullets < 10:
        #     reward -= 1.0 
        #     if distance_to_bullet > 0:  # 確保有補給站
        #         if angle_to_bullet_index == tank_angle_index:  # 朝向補給站
        #             reward += 2.0
        #         else:
        #             reward -= 1.0
        #     if action == 1:
        #         reward += 2.0

        


        #if self.player == "1P":
            #print("obs",obs)

        return reward
    
    def _get_obs(self) -> np.ndarray:
        return self.get_obs(
            self.player,
            self._scene_info
            )
    
    def _get_reward(self, obs: dict, action: int) -> float:
        # Get observation to retrieve the precomputed angle_to_target
        return self.get_reward(obs, action)

    def _is_done(self) -> bool:
        return (
            self._scene_info[self.player]["status"] != "GAME_ALIVE"
            or self._scene_info[self.player]["oil"] == 0
        )

    def _get_commands(self, action: int) -> dict:
        commands = {get_ai_name(id): ["NONE"] for id in range(self.player_num)}
        commands[self.player] = COMMAND[action]
        return commands
    
    def _angle_to_index(self, angle: float) -> int:
        angle = (angle + 360) % 360

        segment_center = (angle + DEGREES_PER_SEGMENT/2) // DEGREES_PER_SEGMENT
        return int(segment_center % (360 // DEGREES_PER_SEGMENT))

    def get_distance(self, x1, y1, x2, y2):
        """
        Calculate the distance between two points
        """
        return ((x1-x2)**2 + (y1-y2)**2)**0.5
    
if __name__ == "__main__":
    env = ResupplyEnv(3, 3, 100, randomize=True, render_mode="human")
    for _ in range(10):
        env.reset()
        for _ in range(1000):
            obs, reward, terminate, _, _ = env.step(env.action_space.sample())  # type: ignore
            print("Observation:", obs)  # 包含 [gun_angle, angle_to_target]
            print("Reward:", reward)    # Reward 基於對準程度
            env.render()
            if terminate:
                break
    env.close()
