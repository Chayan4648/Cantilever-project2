from keras.utils import to_categorical
from caption_model import build_model
from image_preprocessing import preprocess_image
from caption_preprocessing import preprocess_captions
import numpy as np
import pickle

# Load and preprocess the dataset (this part needs to be implemented)

# Example captions and images (replace with actual dataset loading)
captions = ["startseq a cat is sitting on a sofa endseq", "startseq a dog is playing with a ball endseq"]
images = ["path_to_image1.jpg", "path_to_image2.jpg"]

# Preprocess captions
tokenizer, sequences, max_length = preprocess_captions(captions)
vocab_size = len(tokenizer.word_index) + 1

# Preprocess images
X_images = np.array([preprocess_image(img) for img in images])

# Convert captions to categorical
y = np.array([to_categorical(seq, num_classes=vocab_size) for seq in sequences])

# Build the model
model = build_model(vocab_size, max_length)

# Train the model
model.fit([X_images, sequences], y, epochs=10, batch_size=64, validation_split=0.2)

# Save the model and tokenizer
model.save('image_captioning_model.h5')
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
