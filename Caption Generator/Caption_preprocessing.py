import re
import numpy as np
import os
from PIL import Image
from pickle import dump, load

def load_doc(filename):

    # loading the file and returing the data of file 

    file = open(filename,'r')
    text = file.read()
    text = text.split("\n")
    file.close()
    return text 

def caption_grouping(captions):

    # returns a dictionary 
    # key contains image name 
    # values contains 5 captions of the img respectively 
 
    img_captions = dict()
    captions = captions[:-1]
    temp = []
    for cap in captions:
        if len(cap)<20:
            continue
        else:
            temp_key = cap.split('#')
            key = temp_key[0]
            temp_caption = temp_key[-1].split('\t')
            if temp_caption[0]=='4':
                temp.append(temp_caption[-1])
                img_captions[key]=temp
                temp=[]
            else:
                temp.append(temp_caption[-1])
    return img_captions
def caption_preprocessing(img_captions):

    # caption in the dictionary are processed and updated dictionay is returned 

    maxlen = 0
    for key,val in img_captions.items():
        temp=[]
        for sent in val:
            prepro = sent.lower()
            prepro = re.sub('[^a-z0-9 ]','',prepro)
            prepro = [ word for word in prepro.split() if len(word)>1 ]
            prepro = [ word for word in prepro if (word.isalpha())]
            prepro = ' '.join(prepro)
            temp.append(prepro)
            if len(temp)>maxlen:
                maxlen = len(temp)
                
        img_captions[key] = temp
    return (img_captions,maxlen)
 
 
def text_vocabulary(description):

    # returns the vocabulary of captions of all images 

    vocab = set()
    for key in description.keys():
        for d in description[key]:
            vocab.update(d.split())
    return vocab





## driver code

text = load_doc("./Flickr8k_text/Flickr8k.token.txt")
img_captions = caption_grouping(text)
description , max_length= caption_preprocessing(img_captions)
vocab = text_vocabulary(description)
save_description(description,"caption_description.txt")










