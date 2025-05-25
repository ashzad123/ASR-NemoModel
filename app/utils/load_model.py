from nemo.collections.asr.models import EncDecCTCModel
import os

# Make sure the model directory exists
os.makedirs("model", exist_ok=True)

# Load the model from the local .nemo file
model = EncDecCTCModel.restore_from("app/model/stt_hi_conformer_ctc_medium.nemo")

# Export to ONNX
model.export(
    output="app/model/stt_hi_conformer_ctc_medium.onnx",
    onnx_opset_version=14,
    check_trace=False
)