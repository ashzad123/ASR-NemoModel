from nemo.collections.asr.models import EncDecCTCModel

model = EncDecCTCModel.restore_from("app/model/stt_hi_conformer_ctc_medium.nemo")
print("Vocab size:", len(model.decoder.vocabulary))
print(model.decoder.vocabulary)