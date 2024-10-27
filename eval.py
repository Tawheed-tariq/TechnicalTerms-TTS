from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

commit_hash = "1c832de03aeca79cebf3c0242260159cee863a44"
model = SpeechT5ForTextToSpeech.from_pretrained("tawheed-tariq/speecht5_tts", revision=commit_hash)
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

text = "Hello, my name is Tawheed. I am a research engineer at Microsoft."
inputs = processor(text=text, return_tensors="pt")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)

speaker_embeddings = embeddings_dataset[7308]["xvector"]
speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("output.wav", speech.numpy(), samplerate=16000)