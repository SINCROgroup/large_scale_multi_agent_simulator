# This is the a base launcher for LS_MAS (Large scale - Multi agent Simulator)
# Authors 
# 
#
#
#
# Description:
#

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.brownian_motion import brownian_motion

if __name__ == '__main__':

    agents = brownian_motion(None,"config.yaml")
    
    

