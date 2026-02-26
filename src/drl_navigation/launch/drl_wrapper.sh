##!/bin/bash
## Activate virtual environment and run DRL node
#source /home/sandeep/Downloads/gz_ws/drl_venv/bin/activate
#exec python3 /home/sandeep/Downloads/gz_ws/install/drl_navigation/lib/python3.12/site-packages/drl_navigation/drl_navigation_node.py "$@"
#

#!/bin/bash
# Activate virtual environment and run DRL node
source /home/sandeep/Downloads/gz_ws/drl_venv/bin/activate
exec python3 /home/sandeep/Downloads/gz_ws/src/drl_navigation/src/drl_navigation/drl_navigation_node.py "$@"