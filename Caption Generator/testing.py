from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as imgg
from argparse import ArgumentParser

 


def extract_features(filename, model):
        image = Image.open(filename)
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

ap = ArgumentParser()
ap.add_argument('-i', '--image', required=False, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']

path = './Flickr8k_Dataset/Flicker8k_Dataset/2886411666_72d8b12ce4.jpg'
show_img = Image.open(path)
max_length = 32
tokenizer = load(open("./tokenizer.p","rb"))
model = load_model('./models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(path, xception_model)
img = Image.open(path)

description = generate_desc(model, tokenizer, photo, max_length)
description = description.split()
description = description[1:-1]
description = ' '.join(description)
print("\n\n")
print("\n\n")
print(description,"\n\n\n\n")
show_img.show()