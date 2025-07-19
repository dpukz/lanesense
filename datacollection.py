import datetime
import cv2
import math
import os
import time
from opendbc.car.panda_runner import PandaRunner

# output files
curvature_out_file = 'curvature.csv'
speed_out_file     = 'speed.csv'

# set up camera
cam = cv2.VideoCapture(0)
# Optional: ask the camera to run at 30 FPS if it supports it
cam.set(cv2.CAP_PROP_FPS, 30)

# target interval for 30 FPS
FRAME_INTERVAL = 1.0 / 30.0

with PandaRunner() as p:
    params     = p.CI.CP
    WHEELBASE  = params.wheelbase
    STEER_RATIO= params.steerRatio

    # open files once in append mode
    with open(curvature_out_file, "a+", newline="") as cf, \
         open(speed_out_file,     "a+", newline="") as sf:

        # ensure headers
        cf.seek(0)
        if cf.read(1) == "": cf.write("Timestamp,Curvature\n")
        sf.seek(0)
        if sf.read(1) == "": sf.write("Timestamp,Vehicle_Speed\n")

        while True:
            start = time.time()

            state = p.read()
            steer_deg = state.steeringAngleDeg
            curvature = math.tan((steer_deg / STEER_RATIO) * math.pi/180) / WHEELBASE
            speed     = state.vEgo

            ret, img = cam.read()
            ts = datetime.datetime.now().timestamp()

            if ret:
                # format safe filename
                fn = datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
                cv2.imwrite(fn, img)

            # append one line per loop
            cf.write(f"{ts},{curvature}\n")
            sf.write(f"{ts},{speed}\n")

            # throttle to ~30 FPS
            elapsed = time.time() - start
            to_sleep = FRAME_INTERVAL - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
