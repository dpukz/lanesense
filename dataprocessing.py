from panda import Panda
import time
import cantools
import datetime
import cv2
import math
from opendbc.car.toyota import *
from opendbc.car import interfaces, structs

p = Panda()
p.set_safety_mode(Panda.SAFETY_NOOUTPUT)
p.set_can_speed(0, 500000)

WHEELBASE = interfaces.CarInterfaceBase.get_params()

cam = cv2.VideoCapture(0)

db = cantools.database.load_file('my_car.dbc')

curvature_csv = 'Timestamp,Curvature\n'
speed_csv = 'Timestamp,Vehicle_Speed\n'

curvature_out_file = 'curvature.csv'
speed_out_file = 'speed.csv'

try:
    while True:
        msgs = p.can_recv()
        state = structs.CarState
        steer_angle_deg = state.steeringAngleDeg  # in degrees
        curvature = math.tan(steer_angle_deg * math.pi / 180) / WHEELBASE
        speed = state.vEgo
        s, img = cam.read()
        ts = datetime.datetime.now().timestamp()
        if s:
            cv2.imwrite(str(ts) + ".jpg", img)
        curvature_csv += str(ts) + ',' + str(curvature) + '\n'
        speed_csv += str(ts) + ',' + str(speed) + '\n'
finally:
    p.close()