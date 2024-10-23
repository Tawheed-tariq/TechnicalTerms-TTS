from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager

# Model setup
model_name = "tts_models/en/ljspeech/tacotron2-DDC_ph"

# Initialize the ModelManager to download the model
manager = ModelManager()
model_path, config_path, model_item = manager.download_model(model_name)

# Initialize the synthesizer
synth = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path
)

# Input text
text = "Hello, this is a test sentence for TTS."

# Generate waveform from text
wav = synth.tts(text)

# Save the generated audio to a file
synth.save_wav(wav, "output.wav")

print("Speech synthesis complete! Audio saved as output.wav.")
