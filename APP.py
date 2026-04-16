import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# 1. Setup Page Configuration
st.set_page_config(page_title="AI Image Generator", layout="centered")
st.title("🎨 Text-to-Image Generator")

# 2. Load the Model (Cached to avoid reloading on every refresh)
@st.cache_resource
def load_pipeline():
    # Model ID from Hugging Face
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Check if GPU (CUDA) is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    return pipe

# Initialize the pipeline
pipe = load_pipeline()

# 3. User Interface Components
prompt = st.text_input("Enter your description:", placeholder="e.g., A futuristic city in space")

if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating... this may take a moment."):
            # Generate the image
            # Note: CPU generation is much slower than GPU
            result = pipe(prompt).images[0]
            
            # Display result
            st.image(result, caption=f"Prompt: {prompt}", use_container_width=True)
            
            # Save for local download
            result.save("generated_output.png")
            with open("generated_output.png", "rb") as file:
                st.download_button(
                    label="Download Image",
                    data=file,
                    file_name="ai_image.png",
                    mime="image/png"
                )
    else:
        st.warning("Please enter a prompt first!")
