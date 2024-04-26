#!/bin/bash
ps -ef|grep "el_service.py --ip ${ip}"|grep -v grep|awk '{print $2}'|xargs kill -9 

