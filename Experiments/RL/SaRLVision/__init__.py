#-------------------------------------------------------------------------------
# Name:        __init__.py
# Purpose:     Registering environment for SaRLVision.
#
# Author:      Matthias Bartolo <matthias.bartolo@ieee.org>
#
# Created:     February 24, 2024
# Copyright:   (c) Matthias Bartolo 2024-
# Licence:     All rights reserved
#-------------------------------------------------------------------------------
from gymnasium.envs.registration import register

# Registering the environment
register(
    id='DetectionEnv-v0',
    entry_point='SaRLVision.env:DetectionEnv',
    kwargs={"env_config": {'dataset': 'PascalVOC2007_2012Dataset', 'dataset_image_set': 'train', 'dataset_year': '2007+2012'}} # Training environment
)