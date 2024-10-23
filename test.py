import torch
import soundfile as sf
import os
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Config
from safetensors.torch import load_file
from datasets import load_dataset

def get_speaker_embeddings():
    """
    Load speaker embeddings from the CMU Arctic dataset.
    Returns:
        torch.Tensor: Speaker embeddings for voice characteristics.
    """
    print("Loading speaker embeddings from CMU Arctic dataset...")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    
    # Get embeddings for a specific speaker (e.g., 'bdl' - male speaker)
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    return speaker_embeddings

def load_custom_model_and_processor(model_path):
    """
    Load the custom fine-tuned model and processor with error handling.
    
    Args:
        model_path (str): Path to the custom fine-tuned model directory.
    Returns:
        tuple: (processor, model, vocoder)
    """
    try:
        # First check if the directory exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model directory {model_path} does not exist")
            
        # Load processor and model configuration
        print("Loading processor and model configuration...")
        processor = SpeechT5Processor.from_pretrained(model_path)
        config = SpeechT5Config.from_json_file(os.path.join(model_path, 'config.json'))
        
        # Initialize model and vocoder with config
        model = SpeechT5ForTextToSpeech(config)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        
        # Load the safetensors weights
        print("Loading model weights from safetensors...")
        state_dict = load_file(os.path.join(model_path, 'model.safetensors'))
        model.load_state_dict(state_dict)
        
        return processor, model, vocoder
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_speech_with_custom_model(text, model_path, output_path="output.wav", speaker_embeddings=None):
    """
    Generate speech from text using the custom SpeechT5 model.
    
    Args:
        text (str): Input text to convert to speech.
        model_path (str): Path to the custom fine-tuned model directory.
        output_path (str): Path where the output audio will be saved.
        speaker_embeddings (torch.Tensor): Speaker embeddings for voice characteristics.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the custom model and processor
    processor, model, vocoder = load_custom_model_and_processor(model_path)
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Move speaker embeddings to device
    if speaker_embeddings is not None:
        speaker_embeddings = speaker_embeddings.to(device)
    
    # Process the input text
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    # Generate speech
    with torch.no_grad():
        try:
            speech = model.generate_speech(
                input_ids,
                speaker_embeddings=speaker_embeddings,
                vocoder=vocoder
            )
            
            # Save the audio file
            sf.write(output_path, speech.cpu().numpy(), samplerate=16000)
            print(f"Speech generated and saved to {output_path}")
            
        except Exception as e:
            print(f"Error during speech generation: {str(e)}")
            raise

def main():
    # You can use either the final model or a specific checkpoint
    model_path = "./fine_tunned_model"  # Change this to your model path
    
    # Load speaker embeddings
    try:
        speaker_embeddings = get_speaker_embeddings()
        print("Successfully loaded speaker embeddings")
    except Exception as e:
        print(f"Error loading speaker embeddings: {str(e)}")
        return
    
    # Test with different texts
    texts = [
        "The blackmoustachio'd face gazed down from every commanding corner."
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs("generated_audio", exist_ok=True)
    
    try:
        for i, text in enumerate(texts):
            output_path = f"generated_audio/generated_speech_{i+1}.wav"
            print(f"\nGenerating speech for text {i+1}: {text}")
            generate_speech_with_custom_model(text, model_path, output_path, speaker_embeddings)
            
    except Exception as e:
        print(f"\nError during speech generation: {str(e)}")
        print("\nPlease ensure your model directory contains the necessary files.")
        
if __name__ == "__main__":
    main()
