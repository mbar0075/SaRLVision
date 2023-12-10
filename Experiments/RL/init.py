from gymnasium.envs.registration import register
from utils import *
from models import *
from env import *

register(
    id='DetectionEnv-v0',
    entry_point='env:DetectionEnv',
)