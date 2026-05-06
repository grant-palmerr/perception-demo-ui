import onnxruntime as rt

model_path = "./models/od-custom-yolox_s_lite_nuimages_onnx/yolox_s_lite_nuimages.onnx"
artifacts_folder = "./models/od-custom-yolox_s_lite_nuimages_onnx/artifacts"

print("Initializing minimal session...")
try:
    sess = rt.InferenceSession(
        model_path,
        providers=['TIDLExecutionProvider'],
        provider_options=[{'artifacts_folder': artifacts_folder}]
    )
    print("SUCCESS: Subgraph offloaded to DSP.")
except Exception as e:
    print(f"\nCRITICAL FAILURE: {e}")