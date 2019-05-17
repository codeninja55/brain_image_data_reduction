# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# brain_image_data_reduction
# helpers.py
# 
# Attributions: 
# [1] 
# ----------------------------------------------------------------------------------------------------------------------

__author__ = 'Andrew Che <@codeninja55>'
__copyright__ = 'Copyright (C) 2019, Andrew Che <@codeninja55>'
__credits__ = ['']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = 'Andrew Che'
__email__ = 'andrew@neuraldev.io'
__status__ = '{dev_status}'
__date__ = '2019.05.17'

"""helpers.py: 

{Description}
"""
import sys


def prompt_continue(msg='\n\nDo you want to continue to training'):
    ans = input('{} Y/n '.format(msg))
    if len(ans) != 0:
        if ans.lower() != "y":
            print('Exiting...')
            sys.exit(0)
    return True


