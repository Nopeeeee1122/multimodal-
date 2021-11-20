import copy
import numpy as np
from transformers import *


def re_tokenize_sentence(args, flag):
    """
        convert original sentence to token sequence, such as [101, 27623, 7873, ...]

    Args:
        flag ([type]): dataset 
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokenized_texts = []
    original_texts = flag['original_post']
    for sentence in original_texts:
        tokenized_text = tokenizer.encode(sentence)
        tokenized_texts.append(tokenized_text)
    flag['post_text'] = tokenized_texts
    
    # entity 
    tokenized_ents = []
    entity = flag['entity']
    for ent in entity:
        tokenized_ent = tokenizer.encode(ent)
        tokenized_ents.append(tokenized_ent)
    flag['entity_token'] = tokenized_ents


def get_all_text(train, validate, test):
    all_text = list(train['post_text']) + list(validate['post_text']) + list(
        test['post_text'])
    return all_text

def align_data(flag, args):
    """
       to align data so that they have the same length

    Args:
        flag ([type]): [description]
        args ([type]): [description]
    """
    text = []
    mask = []
    for sentence in flag['post_text']:
        sen_embedding = []
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word)

        # align word embedding
        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)

        text.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
    flag['post_text'] = text
    flag['mask'] = mask

    entity = []
    ent_mask = []
    for ent in flag['entity_token']:
        ent_embedding = []
        mask_ent = np.zeros(args.ent_len, dtype=np.float32)
        mask_ent[:len(ent)] = 1.0
        for _, w in enumerate(ent):
            ent_embedding.append(w)

        while len(ent_embedding) < args.ent_len:
            ent_embedding.append(0)

        entity.append(copy.deepcopy(ent_embedding))
        ent_mask.append(copy.deepcopy(mask_ent))

    flag['entity_token'] = entity
    flag['mask_ent'] = ent_mask


def word2vec(args, post, word_id_map, W):
    word_embedding = []
    mask = []
    # length = []

    for sentence in post:
        sen_embedding = []
        seq_len = len(sentence) - 1
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word_id_map[word])

        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)

        word_embedding.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
        # length.append(seq_len)
    return word_embedding, mask

