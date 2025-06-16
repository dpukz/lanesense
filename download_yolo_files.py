import urllib.request

files = [
    ("yolov4-tiny.cfg",
     "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"),
    ("yolov4-tiny.weights",
     "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"),
    ("coco.names",
     "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"),
]

for fname, url in files:
    print(f"Downloading {fname} …")
    urllib.request.urlretrieve(url, fname)

print("All files downloaded.")
