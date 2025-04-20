from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, BlipProcessor, BlipForQuestionAnswering
import torch
from PIL import Image
import numpy as np

caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)
vqa_model.to(device)

max_length = 16
num_beams = 2
caption_gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def generate_caption(image_path):
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        img = np.array(img)

    pixel_values = caption_feature_extractor(images=img, return_tensors="pt").pixel_values.to(device)

    output_ids = caption_model.generate(pixel_values, **caption_gen_kwargs)

    caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()

def answer_question(image_path, question):
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        img = np.array(img)  

    inputs = vqa_processor(images=img, text=question, return_tensors="pt").to(device)

    output = vqa_model.generate(**inputs)
    answer = vqa_processor.decode(output[0], skip_special_tokens=True)
    return answer

image_path = 'upload/download (3).jpg'

caption = generate_caption(image_path)
print(f"Generated Caption: {caption}") 

question = "What is the kid t-shirt colour?"
answer = answer_question(image_path, question) 
print(f"Question: {question}")
print(f"Answer: {answer}")