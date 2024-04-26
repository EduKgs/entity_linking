#!/bin/bash
source /data02/gob/virtua_env/python3.7/bin/activate
nohup python -u el_service.py --ip '0.0.0.0' --port '1080' > service.log 2>&1 &
