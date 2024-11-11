import numpy as np
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

"""
Gymnasium Custom environment: MalinHumanoidEnv by Malin Barg
--> Environment to change observation and action space of existing Humanoid-v4 environment
--> 11.03.2024
"""


class MalinHumanoidEnv(HumanoidEnv):
    def __init__(self, **kwargs):
        super(MalinHumanoidEnv, self).__init__(**kwargs)

        # Define custom observation space shape
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64
        )

        # initialise MujocoEnv with changed xml file
        MujocoEnv.__init__(
            self,
            "humanoid.xml",
            5,
            observation_space=self.observation_space,
            **kwargs,
        )


"""
Changing the observation space shape and the xml file also automatically changes the action space.

Original Humanoid 
Observation Space: Box(-inf, inf, (376,), float64)
Action Space :Box(-0.4, 0.4, (17,), float32)

Humanoid without Arms
Observation Space: Box(-inf, inf, (270,), float64)
Action Space :Box(-0.4, 0.4, (11,), float32)

"""
