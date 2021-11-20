import re
from spacy import load as spacy_load
from spikex.wikigraph import load as wg_load
from spikex.pipes import WikiPageX
import pickle

from transformers import file_utils


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
    return string.strip().lower()



def get_text_dict():
    pre_path = "Data/weibo/tweets/"
    file_list = [pre_path + "test_nonrumor.txt", pre_path + "test_rumor.txt", \
                pre_path + "train_nonrumor.txt", pre_path + "train_rumor.txt"]


    for k, fp in enumerate(file_list):
          
          f = open(fp, "r")
          
          twitter_id = 0
          id_list = []
          news_list = []
          for i, l in enumerate(f.readlines()):
              
              if(i+ 1 ) % 3 == 1:
                    twitter_id = l.split('|')[0]
                    id_list.append(twitter_id)
              
              if (i + 1) %3 == 0:
                    l = clean_str_sst(l)
                    news_list.append(l)
          data_dict = dict(zip(id_list, news_list))
          pick_path = fp.replace('rumor.txt', 'rumor_ent.pkl')
          generate_entity(data_dict, pick_path)

def generate_entity(data_dict, output_path):

    output_text = open(output_path, 'wb')
    output_dict = dict()
    nlp = spacy_load("zh_core_web_trf")

    for k, v in data_dict.items():
      entity = dict()
      doc = nlp(v)
      for ent in doc.ents:
          entity[ent.text] = ent.label_
          print(k, ent.text, "'s entity is", ent.label_)
      output_dict[k] = entity

    pickle.dump(output_dict, output_text)

def main():
  get_text_dict()

if __name__ == '__main__':
    main()
    
