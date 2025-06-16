import cv2, onnxruntime as ort, numpy as np, time

sess = ort.InferenceSession("supercombo.onnx",
                            providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name

def prep(img):
    img = cv2.resize(img, (512, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) / 255.0).transpose(2,0,1)[None]
    return img

cap = cv2.VideoCapture(0)            # default webcam

while True:
    ok, frame = cap.read()
    if not ok:
        break

    pred = sess.run(None, {inp_name: prep(frame)})[0][0]  # (3, 33) path
    xs = pred[0, :30] * 3.2 + frame.shape[1] / 2          # crude warp
    ys = np.linspace(100, frame.shape[0]-50, 30)

    for x, y in zip(xs, ys):
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

    cv2.imshow("OpenPilot path demo  –  press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
