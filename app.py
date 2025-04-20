#streamlit

import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, BlipProcessor, BlipForQuestionAnswering
import torch
from PIL import Image
import numpy as np

# Load models
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)
vqa_model.to(device)

caption_gen_kwargs = {"max_length": 16, "num_beams": 2}

def generate_caption(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    img_np = np.array(image)

    pixel_values = caption_feature_extractor(images=img_np, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        output_ids = caption_model.generate(pixel_values, **caption_gen_kwargs)

    caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()

def answer_question(image: Image.Image, question: str):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    img_np = np.array(image)

    inputs = vqa_processor(images=img_np, text=question, return_tensors="pt").to(device)

    with torch.no_grad():
        output = vqa_model.generate(**inputs)

    answer = vqa_processor.decode(output[0], skip_special_tokens=True)
    return answer.strip()

# Streamlit UI
st.title(" Image Caption Generator & Visual Question Answering")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        caption = generate_caption(image)
        st.success(f" Caption: {caption}")

    question = st.text_input("Ask a question about the image:")

    if question:
        answer = answer_question(image, question)
        st.info(f"Answer: {answer}")
