import streamlit as st
import os
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
import soundfile as sf
import subprocess
import pysrt
import tempfile
from datetime import datetime

def process_srt(file, output_dir, description, model_size):
    # Create a temporary file to save the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix='.srt') as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name

    # Load the SRT file
    subs = pysrt.open(temp_file_path)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up Parler-TTS
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_id = f"parler-tts/parler-tts-{model_size}-v1"
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    class CudaMemoryManager:
        def __enter__(self):
            torch.cuda.empty_cache()

        def __exit__(self, exc_type, exc_val, exc_tb):
            torch.cuda.empty_cache()

    def generate_audio(text):
        with CudaMemoryManager():
            inputs = tokenizer([description], return_tensors="pt", padding=True).to(device)
            prompt = tokenizer([text], return_tensors="pt", padding=True).to(device)
            
            set_seed(0)
            generation = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                prompt_input_ids=prompt.input_ids,
                prompt_attention_mask=prompt.attention_mask,
                do_sample=True,
                return_dict_in_generate=True,
            )
            
            audio = generation.sequences[0, :generation.audios_length[0]]
            
            # Clear unnecessary tensors
            del inputs, prompt, generation
        
        return audio

    for i, sub in enumerate(subs):
        audio = generate_audio(sub.text)
        output_file = os.path.join(output_dir, f"{i+1:04d}.wav")
        sf.write(output_file, audio.cpu().numpy().squeeze(), feature_extractor.sampling_rate)
        
        # Update progress
        progress_bar.progress((i + 1) / len(subs))

    # Move model to CPU to free up GPU memory
    model.cpu()
    torch.cuda.empty_cache()

def process_text(text, output_dir, description, model_size):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up Parler-TTS
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_id = f"parler-tts/parler-tts-{model_size}-v1"
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    class CudaMemoryManager:
        def __enter__(self):
            torch.cuda.empty_cache()

        def __exit__(self, exc_type, exc_val, exc_tb):
            torch.cuda.empty_cache()

    with CudaMemoryManager():
        inputs = tokenizer([description], return_tensors="pt", padding=True).to(device)
        prompt = tokenizer([text], return_tensors="pt", padding=True).to(device)
        
        set_seed(0)
        generation = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
            do_sample=True,
            return_dict_in_generate=True,
        )
        
        audio = generation.sequences[0, :generation.audios_length[0]]
        
        # Clear unnecessary tensors
        del inputs, prompt, generation
    
    output_file = os.path.join(output_dir, "0001.wav")
    sf.write(output_file, audio.cpu().numpy().squeeze(), feature_extractor.sampling_rate)
    
    # Move model to CPU to free up GPU memory
    model.cpu()
    torch.cuda.empty_cache()
    
    # Update progress
    progress_bar.progress(1.0)

def combine_audio_files(input_dir, output_file):
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    # Sort the files to ensure correct order
    wav_files.sort()
    
    # Create a list file for ffmpeg
    list_file = 'file_list.txt'
    with open(list_file, 'w') as f:
        for wav_file in wav_files:
            f.write(f"file '{os.path.join(input_dir, wav_file)}'\n")
    
    # Construct the ffmpeg command
    ffmpeg_command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file,
        '-c:a', 'libmp3lame',
        '-b:a', '128k',
        output_file
    ]
    
    # Execute the ffmpeg command
    subprocess.run(ffmpeg_command, check=True)
    
    # Remove the temporary list file
    os.remove(list_file)

# Streamlit UI
st.title("SRT/Text to Audio Converter")

st.markdown("""
## How to Create a Good Description

1. **Voice Quality**: Use "very clear audio" for high-quality audio or "very noisy audio" for background noise.
2. **Prosody Control**: Use punctuation (e.g., commas) to add small breaks in speech.
3. **Speech Features**: Directly control gender, speaking rate, pitch, and reverberation through the prompt.
4. **Available Voices**: Jon, Lea, Gary, Jenna, Mike, Laura (use these names in your description for consistency)

**Example**: "Jon's voice is very clear audio, with a slow speaking rate and low pitch. The recording has minimal reverberation."
""")

input_type = st.radio("Choose input type:", ["subtitle", "text"])

if input_type == "subtitle":
    uploaded_file = st.file_uploader("Choose an SRT file", type="srt")
else:
    text_input = st.text_area("Enter your text:", height=200)

default_description = "Jon has a clear and natural voice with neutral intonation."
description = st.text_input("Enter voice description:", value=default_description)

model_size = st.selectbox("Choose model size:", ["mini", "large"])

if st.button("RUN"):
    # Generate unique output filename based on current date and time
    current_time = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
    output_file = os.path.join("output", f"{current_time}.wav")
    output_dir = os.path.join("output", current_time)
    
    progress_bar = st.progress(0)
    
    if input_type == "subtitle" and uploaded_file is not None:
        st.write("Processing subtitles...")
        process_srt(uploaded_file, output_dir, description, model_size)
    elif input_type == "text" and text_input:
        st.write("Processing text...")
        process_text(text_input, output_dir, description, model_size)
    else:
        st.error("Please provide input (either upload an SRT file or enter text).")
        st.stop()
    
    # Combine audio files
    st.write("Combining audio files...")
    combine_audio_files(output_dir, output_file)
    
    st.write("Processing complete!")
    
    # Check if the output file exists before trying to play it
    if os.path.exists(output_file):
        st.audio(output_file)
    else:
        st.error(f"Error: The output file {output_file} was not created. Please check the logs for more information.")

# Add a note about required libraries
st.sidebar.markdown("""
## Required Libraries
- streamlit
- torch
- parler_tts
- transformers
- soundfile
- pysrt
- ffmpeg (system installation)

Install with:
```
pip install streamlit torch parler_tts transformers soundfile pysrt
```
And install ffmpeg on your system.
""")