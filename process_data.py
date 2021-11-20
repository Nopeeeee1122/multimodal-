import numpy as np
import pickle
import os
from src.process_twitter import clean_text, clean_str_sst
import re

def remove_urls (vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    vTEXT = re.sub(
        u'['
        u'\U0001F300-\U0001F64F'
        u'\U0001F680-\U0001F6FF'
        u'\u2600-\u26FF\u2700-\u27BF]+', r" ", vTEXT)
    return(vTEXT)
    
def read_post(data_path):
    data_dict = {}
    with open(data_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i==0: continue
            post_id, text = line.split('\t')[0], line.split('\t')[1]
            data_dict[post_id] = remove_urls(text)
    return data_dict

def process(data_path, output_path):
    data_dict = read_post(data_path)
    pick_data = open(output_path, 'wb')
    pickle.dump(data_dict, pick_data)
    print("process done!")

if __name__ == '__main__':
    data_path ='Data/twitter/train_posts.txt'
    output_path = 'Data/twitter/cleaned_train_text.pkl'
    process(data_path, output_path)
    