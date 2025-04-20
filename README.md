# Image Captioning & Visual Question Answering

This project uses Hugging Face Transformers to perform image captioning and visual question answering (VQA) on input images. It uses the `vit-gpt2` model for captions and `blip-vqa` for answering questions about images.

---

## Features

- Generate a caption for a given image
- Ask questions about an image and get intelligent answers
- Uses pre-trained Hugging Face models

---

## Models Used

- **Image Captioning**: [`nlpconnect/vit-gpt2-image-captioning`](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)
- **Visual Question Answering**: [`Salesforce/blip-vqa-small`](https://huggingface.co/Salesforce/blip-vqa-small)

---

## Requirements

Install dependencies:

```bash
pip install transformers torch torchvision pillow

----------------------
For Windows users facing DLL issues, use the CPU-only version of PyTorch:

pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2 --index-u