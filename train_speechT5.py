import os
import pandas as pd
import librosa
import numpy as np
import torch
from datasets import Dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, Trainer, TrainingArguments

class CustomDataCollator:
    def __init__(self, processor, target_audio_length=80000):  # Set target audio length
        self.processor = processor
        self.target_audio_length = target_audio_length

    def __call__(self, features):
        # Process text inputs
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        # Pad input ids and attention mask
        max_length = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_attention_mask = []

        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_length - len(ids)
            padded_input_ids.append(ids + [self.processor.tokenizer.pad_token_id] * padding_length)
            padded_attention_mask.append(mask + [0] * padding_length)

        # Convert to tensors
        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)  # Keep input IDs as Long
        attention_mask_tensor = torch.tensor(padded_attention_mask, dtype=torch.long)  # Keep attention mask as Long

        # Process audio features
        speech_arrays = [np.array(f["labels"]) for f in features]

        # Pad audio features to the specified target length
        padded_speech = []
        for audio in speech_arrays:
            if len(audio) > self.target_audio_length:
                audio = audio[:self.target_audio_length]  # Truncate if too long
            else:
                audio = np.pad(audio, (0, self.target_audio_length - len(audio)), mode='constant')
            padded_speech.append(audio)

        # Convert to tensor and ensure the dtype is Float
        speech_tensor = torch.tensor(padded_speech, dtype=torch.float32)  # Ensure it's float32

        # Ensure the shape of audio matches what the model expects
        speech_tensor = speech_tensor.unsqueeze(1)  # Adding an additional dimension

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": speech_tensor,
        }

def load_audio(file_path, dataset_dir, max_duration=10.0):
    """Load and process audio file with duration limit"""
    full_path = os.path.join(dataset_dir, file_path)
    if not os.path.exists(full_path):
        print(f"Audio file does not exist: {full_path}")
        return None
    
    try:
        # Load audio with specified duration limit
        speech_array, sampling_rate = librosa.load(full_path, sr=16000, duration=max_duration)
        if speech_array is not None and len(speech_array) > 0:
            speech_array = librosa.util.normalize(speech_array).astype(np.float32)  # Ensure float32
            # Ensure consistent length (10 seconds = 160000 samples at 16kHz)
            target_length = int(max_duration * 16000)
            if len(speech_array) > target_length:
                speech_array = speech_array[:target_length]
            elif len(speech_array) < target_length:
                speech_array = np.pad(speech_array, (0, target_length - len(speech_array)), mode='constant')
            return speech_array.tolist()
        else:
            print(f"Loaded audio is empty: {full_path}")
            return None
    except Exception as e:
        print(f"Error loading audio from {full_path}: {e}")
        return None

def prepare_dataset(df, dataset_dir, processor):
    """Prepare dataset with both audio and text data"""
    processed_data = []
    
    print("Processing audio files...")
    for idx, row in df.iterrows():
        audio = load_audio(row['Audio Path'], dataset_dir)
        if audio is not None:
            # Process the text
            text_inputs = processor(
                text=row['Text'],
                return_tensors="pt",
                padding=True,
                max_length=256,
                truncation=True
            )
            
            processed_item = {
                'input_ids': text_inputs['input_ids'][0].tolist(),
                'attention_mask': text_inputs['attention_mask'][0].tolist(),
                'labels': audio,
                'text': row['Text']
            }
            
            processed_data.append(processed_item)
            
        if idx % 100 == 0:
            print(f"Processed {idx} files...")
    
    if not processed_data:
        raise ValueError("No data was successfully processed")
    
    print(f"Successfully processed {len(processed_data)} files")
    dataset = Dataset.from_list(processed_data)
    return dataset

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model components
    print("Loading model components...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    model.to(device)

    # Load metadata
    print("Loading metadata...")
    dataset_dir = 'dataset'
    metadata_path = os.path.join(dataset_dir, 'metadata.csv')
    
    df = pd.read_csv(metadata_path)
    df['Audio Path'] = df['Audio Path'].apply(lambda x: os.path.join('wavs', x.strip()))

    # Prepare dataset
    print("\nPreparing dataset...")
    dataset = prepare_dataset(df, dataset_dir, processor)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        save_steps=500,
        eval_steps=500,
        logging_dir="./logs",
        learning_rate=5e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CustomDataCollator(processor),
    )

    # Start training
    print("\nStarting training...")
    trainer.train()

    # Save the model
    print("\nSaving model...")
    model.save_pretrained("./fine_tuned_model")
    processor.save_pretrained("./fine_tuned_model")

if __name__ == "__main__":
    main()
