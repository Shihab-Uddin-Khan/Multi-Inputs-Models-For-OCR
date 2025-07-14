# 🧠 Multi-Inputs-Models-For-OCR
This project showcases a deep learning pipeline for Optical Character Recognition (OCR) using multi-modal inputs—combining image data with categorical metadata (insurance type)—to identify and classify document IDs. Built using PyTorch, the model is tailored for DigiNsure Inc.'s claim digitization initiative, enabling intelligent identification of primary and secondary IDs in scanned insurance documents.
# 🚀 Project Highlight:
- Multi-Input Architecture: Combines CNN-based image encoding with insurance type vectors using fully connected layers.
- Custom Dataset Loader: Loads and visualizes annotated document images with mapped labels.
- Efficient Training Loop: Implements adaptive optimization via Adam and logs training losses across 20 epochs.
- Data Modalities Used:
- 🖼️ Scanned grayscale document images (64×64)
- 📊 One-hot encoded insurance type: home, life, auto, health, other
# 🏗️ Model Architecture
- Image Path: 1-channel convolution → ELU activation → flatten → dense embedding
- Type Path: Linear transformation → ELU activation
- Fusion: Concatenated embeddings fed into MLP for label classification (primary/secondary ID)
