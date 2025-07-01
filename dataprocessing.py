from panda import Panda
import time
import cantools
import datetime
import cv2
import math

p = Panda()
p.set_safety_mode(Panda.SAFETY_NOOUTPUT)
p.set_can_speed(0, 500000)

WHEELBASE = 1

cam = cv2.VideoCapture(0)

db = cantools.database.load_file('my_car.dbc')

curvature_csv = 'Timestamp,Curvature\n'
speed_csv = 'Timestamp,Vehicle_Speed\n'

curvature_out_file = 'curvature.csv'
speed_out_file = 'speed.csv'

try:
    while True:
        msgs = p.can_recv()
        ts = time.time()
        for arb_id, bus, data, flags in msgs:
            print(f"{ts:.4f} ID=0x{arb_id:x} BUS={bus} DATA={data.hex()}")
            # msg_id, data are from p.can_recv()
            try:
                signals = db.decode_message(arb_id, data)
                # If in opendbc the steering signal is named “STEER_ANGLE”:
                steer_angle_deg = signals['STEER_ANGLE']   # in degrees
                curvature = math.tan(steer_angle_deg * math.pi / 180) / WHEELBASE
                speed = signals['VEHICLE_SPEED']
                s, img = cam.read()
                ts = datetime.datetime.now().timestamp()
                if s:
                    cv2.imwrite(str(ts) + ".jpg", img)
                curvature_csv += str(ts) + ',' + str(curvature) + '\n'
                speed_csv += str(ts) + ',' + str(speed) + '\n'
            except KeyError:
                continue   # this ID isn’t in the DBC
        time.sleep(0.05)
finally:
    p.close()