from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, BlipProcessor, BlipForQuestionAnswering
import torch
from PIL import Image
import numpy as np

                                                                                                                                                                                                        
caption_model = VisionEncoderDecoderModel.from_pretrained("google/vit-base-patch16-224-in21k") 
caption_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
caption_tokenizer = AutoTokenizer.from_pretrained("gpt2")  

vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-small") 
vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-small")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)
vqa_model.to(device)

max_length = 16
num_beams = 2 
caption_gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def preprocess_image(image_path, target_size=(224, 224)):
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        img = img.resize(target_size)
    return np.array(img)

def generate_caption(image_path):
    img = preprocess_image(image_path)
    pixel_values = caption_feature_extractor(images=img, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        output_ids = caption_model.generate(pixel_values, **caption_gen_kwargs)

    caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()

def answer_question(image_path, question):
    img = preprocess_image(image_path)
    inputs = vqa_processor(images=img, text=question, return_tensors="pt").to(device)

    with torch.no_grad():
        output = vqa_model.generate(**inputs)

    answer = vqa_processor.decode(output[0], skip_special_tokens=True)
    return answer

image_path = '/content/download (3).jpg'

caption = generate_caption(image_path)
print(f"Generated Caption: {caption}")

question = "What is the kid doing?"
answer = answer_question(image_path, question)
print(f"Question: {question}")
print(f"Answer: {answer}")



#python -m venv myenv
#myenv\Scripts\activate
