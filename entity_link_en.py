from numpy.lib.function_base import append
from spacy import load as spacy_load
from spikex.wikigraph import load as wg_load
from spikex.pipes import WikiPageX
import pickle


def get_text_dict(flag):
    pre_path = "Data/twitter/"
    if flag == 'train':
        text_dict = pickle.load(open(pre_path + 'cleaned_train_text.pkl',
                                     'rb'))
    elif flag == 'test':
        text_dict = pickle.load(open(pre_path + 'cleaned_test_text.pkl', 'rb'))
    else:
        text_dict = {}
        print('Error of getting text dict, because of the wrong flag')
    return text_dict


def main():
    text_dict = get_text_dict('test')

    output_text = open('Data/twitter/cleaned_test_text_entity.pkl', 'wb')

    nlp = spacy_load("en_core_web_lg")

    data = {}
    for key, value in text_dict.items():
      doc = nlp(value)
      entity = dict()
      for ent in doc.ents:
          entity[ent.text] = ent.label_
          print(key, ent.text, "'s entity is", ent.label_)
      
      data[key] = entity
    pickle.dump(data, output_text)

if __name__ == '__main__':
    main()
    
