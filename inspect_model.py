import onnxruntime as ort

sess = ort.InferenceSession("supercombo.onnx")
print("Inputs:")
for inp in sess.get_inputs():
    print(f"  {inp.name}: {inp.shape}")
print("Outputs:")
for out in sess.get_outputs():
    print(f"  {out.name}: {out.shape}")