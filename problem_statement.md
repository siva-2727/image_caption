Problem Statement: 

Image Captioning and Visual Question Answering Using Transformers
In today’s visually driven world, extracting meaningful information from images plays a crucial role across industries such as healthcare, education, surveillance, and e-commerce. This project aims to develop a system capable of generating descriptive captions for images and answering user-posed questions based on image content using state-of-the-art transformer models.

-Objective
To create an AI-powered application that:

Automatically generates a natural language caption describing the content of a given image.

Allows users to ask questions about the image and receive accurate answers derived from its visual features.

- Key Features
Utilizes the ViT-GPT2 model for image captioning (nlpconnect/vit-gpt2-image-captioning).

Employs the BLIP model (Salesforce/blip-vqa-base) for visual question answering.

Provides a Streamlit-based UI for easy interaction with the models — upload an image, generate captions, and ask questions.

Supports both CPU and GPU execution for flexibility and performance.

- Functionality
Input: Any JPEG/PNG image uploaded by the user.

Output:

A caption summarizing the visual content.

An answer to a user-provided question about the image.

- Use Cases
Educational tools for visually describing images to children or language learners.

Accessibility aids for visually impaired users.

Automated tagging for media/content management systems.

Customer support in e-commerce, helping bots understand product images.

- Expected Outcome
An interactive system that combines computer vision and natural language processing to interpret and describe images, making them more accessible and interactive.