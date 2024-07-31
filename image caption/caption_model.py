from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.applications.vgg16 import VGG16

def build_model(vocab_size, max_length):
    # Load pre-trained VGG16 model for image feature extraction
    image_model = VGG16(include_top=False, weights='imagenet')
    image_model = Model(inputs=image_model.input, outputs=image_model.layers[-1].output)

    # Captioning model
    embedding_dim = 256

    caption_model = Sequential()
    caption_model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    caption_model.add(LSTM(256))
    caption_model.add(Dropout(0.5))
    caption_model.add(Dense(256, activation='relu'))

    # Merge image and caption models
    image_input = Input(shape=(7, 7, 512))
    image_features = Dense(256, activation='relu')(image_model.output)
    combined = add([image_features, caption_model.output])

    final_output = Dense(vocab_size, activation='softmax')(combined)
    model = Model(inputs=[image_model.input, caption_model.input], outputs=final_output)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
