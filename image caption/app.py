from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from image_preprocessing import preprocess_image
from caption_model import build_model
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the tokenizer and model
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model('image_captioning_model.h5')
max_length = 34  # Set this to the maximum length of the sequences used during training

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the image
        image = preprocess_image(file_path)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Generate caption
        caption = generate_caption(model, tokenizer, image, max_length)
        
        return render_template('result.html', caption=caption, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
