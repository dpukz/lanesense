import datetime
import cv2
import math
import os
from opendbc.car.panda_runner import PandaRunner

curvature_out_file = 'curvature.csv' # change the directory to something on your actual drive
speed_out_file = 'speed.csv' # change this directory to something on your actual drive

cam = cv2.VideoCapture(0)

curvature_header = 'Timestamp,Curvature\n' # CSV headers
speed_header = 'Timestamp,Vehicle_Speed\n'

curvature_csv = '' # actual data collected goes here; needed to accomodate files being appended to
speed_csv = ''

with PandaRunner() as p: # interface with car

    # Finds car-specific constants
    params = p.CI.CP
    WHEELBASE = params.wheelbase
    STEER_RATIO = params.steerRatio

    # Actual data collection
    while True:
        state = p.read()
        steer_angle_deg = state.steeringAngleDeg  # steering wheel angle (in degrees)
        curvature = math.tan((steer_angle_deg / STEER_RATIO) * (math.pi / 180)) / WHEELBASE # converts steering wheel angle to curvature
        speed = state.vEgo # velocity
        s, img = cam.read() # camera frame
        ts = datetime.datetime.now().timestamp() # time
        if s:
            cv2.imwrite(str(ts) + ".jpg", img) # saves camera frame if exists
        curvature_csv += str(ts) + ',' + str(curvature) + '\n'
        speed_csv += str(ts) + ',' + str(speed) + '\n'

# Writes speed and curvature to files
with open(curvature_out_file, "a+") as curvature_file:
    curvature_file.seek(0)
    for curvature_line in curvature_file: # already used file
        if curvature_line == curvature_header:
            curvature_file.seek(0, os.SEEK_END)
            curvature_file.write(curvature_csv)
            curvature_file.close()
            break
        else: # new file
            curvature_file.seek(0, os.SEEK_END)
            curvature_file.write(curvature_header)
            curvature_file.write(curvature_csv)
            curvature_file.close()
            break

with open(speed_out_file, "a+") as speed_file:
    speed_file.seek(0)
    for speed_line in speed_file: # already used file
        if speed_line == speed_header:
            speed_file.seek(0, os.SEEK_END)
            speed_file.write(speed_csv)
            speed_file.close()
            break
        else: # new file
            speed_file.seek(0, os.SEEK_END)
            speed_file.write(speed_header)
            speed_file.write(speed_csv)
            speed_file.close()
            break