import cv2
from hikvisionapi import Client


import numpy as np
import HKIPcamera


ip_address = 'https://172.29.239.9'
username = 'admin'
password = 'Purvar123'

client = Client(ip_address,username,password)
client.login(ip_address, username, password)
client.connect()
stream = client.get_stream()
