import numpy as np
import argparse
import time, os
# import random
import src.process_twitter as process_twitter_data
import src.process_data_weibo2 as process_weibo_data
import copy
import torch
import pickle as pickle
from random import sample
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from sklearn import metrics
from transformers import *

from src.dataset.rumor_data import Rumor_Data
from src.model.model import Twitter_Model
from src.utils.utils import to_var, to_np
from src.dataset.preprocess import re_tokenize_sentence, get_all_text, align_data

import warnings

warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def training(model, train_loader, val_loader, test_loader):
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        list(model.parameters())),
                                 lr=args.learning_rate,
                                 weight_decay=0.1)

    print("loader size " + str(len(train_loader)))

    print('training model')
    best_acc = 0.0
    # Train the Model
    model.train()
    for epoch in range(args.num_epochs):

        p = float(epoch) / 100
        lr = 0.001 / (1. + 10 * p)**0.75

        optimizer.lr = lr
        cost_vector = []
        class_cost_vector = []
        acc_vector = []

        for i, (train_data, train_labels,
                event_labels) in enumerate(train_loader):
            train_text, train_image, train_mask, train_entity, train_labels, event_labels  = \
                to_var(train_data[0]), to_var(train_data[1]), to_var(train_data[2]), to_var(train_data[3]), \
                to_var(train_labels), to_var(event_labels)
            # Forward + Backward + Optimize
            optimizer.zero_grad()

            class_outputs = model(train_text, train_image,
                                                  train_mask, train_entity)

            loss = criterion(class_outputs, train_labels)
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(class_outputs, 1)

            accuracy = (train_labels == argmax.squeeze()).float().mean()

            cost_vector.append(loss.item())
            acc_vector.append(accuracy.item())


        best_val_acc = val(model, val_loader, criterion)

        print('Epoch [%d/%d],  Loss: %.4f,  Train_Acc: %.4f , Validate_Acc: %.4f.'\
            % (epoch + 1, args.num_epochs, np.mean(cost_vector), np.mean(acc_vector), best_val_acc))

        if best_val_acc > best_acc:
            best_acc = best_val_acc
            best_validate_dir = os.path.join(args.output_file , str(epoch + 1) + '.pkl')
            torch.save(model.state_dict(), best_validate_dir)

    best_validate_dir = 'Data/twitter/RESULT_text_only/40.pkl'
    testing(model, best_validate_dir, test_loader)

def val(model, val_loader, criterion):
    
    valid_acc_vector = []
    vali_cost_vector = []
    validate_acc_vector_temp = []

    model.eval()
    for i, (validate_data, validate_labels,
            event_labels) in enumerate(val_loader):
        validate_text, validate_image, validate_mask, validate_entity, validate_labels, event_labels = \
            to_var(validate_data[0]), to_var(validate_data[1]), to_var(validate_data[2]), to_var(validate_data[3]), \
            to_var(validate_labels), to_var(event_labels)
        validate_outputs = model(validate_text, validate_image,
                                                 validate_mask,
                                                 validate_entity)
        _, validate_argmax = torch.max(validate_outputs, 1)
        vali_loss = criterion(validate_outputs, validate_labels)
        validate_accuracy = (
            validate_labels == validate_argmax.squeeze()).float().mean()
        vali_cost_vector.append(vali_loss.item())
        validate_acc_vector_temp.append(validate_accuracy.item())

    validate_acc = np.mean(validate_acc_vector_temp)
    valid_acc_vector.append(validate_acc)

    best_validate_acc = validate_acc
    if not os.path.exists(args.output_file):
        os.mkdir(args.output_file)

    return best_validate_acc

def testing(model, best_validate_dir, test_loader):
    print('testing model')
    model.load_state_dict(torch.load(best_validate_dir))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
        test_text, test_image, test_mask, test_entity, test_labels = to_var(
            test_data[0]), to_var(test_data[1]), to_var(test_data[2]), to_var(
                test_data[3]), to_var(test_labels)
        test_outputs = model(test_text, test_image, test_mask,
                                             test_entity)
        _, test_argmax = torch.max(test_outputs, 1)
        if i == 0:
            test_score = to_np(test_outputs.squeeze())
            test_pred = to_np(test_argmax.squeeze())
            test_true = to_np(test_labels.squeeze())
        else:
            test_score = np.concatenate((test_score, to_np(test_outputs)),
                                        axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax)), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels)), axis=0)

    # compute metrics
    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    test_precision = metrics.precision_score(test_true,
                                             test_pred,
                                             average='macro')
    test_recall = metrics.recall_score(test_true, test_pred, average='macro')
    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true,
                                        test_score_convert,
                                        average='macro')

    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    print("Classification Acc: %.4f, AUC-ROC: %.4f" %
          (test_accuracy, test_aucroc))
    print("Classification report:\n%s\n" %
          (metrics.classification_report(test_true, test_pred)))
    print("Classification confusion matrix:\n%s\n" % (test_confusion_matrix))


def main(args):
    print(args)
    if not os.path.exists(args.output_file):
        os.mkdir(args.output_file)
    print('loading data')
    train, validation, test = load_data(args)
    test_id = test['post_id']

    train_dataset = Rumor_Data(train)

    validate_dataset = Rumor_Data(validation)

    test_dataset = Rumor_Data(test)

    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    print('building model')
    model = Twitter_Model(args)

    if torch.cuda.is_available():
        print("CUDA")
        model.cuda()
    
    training(model, train_loader, validate_loader, test_loader)

def load_data(args):
    """
       load train data and test data

    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
 
    train, validate, test, _ = process_twitter_data.get_data(args.text_only)
    re_tokenize_sentence(args, train)
    re_tokenize_sentence(args, validate)
    re_tokenize_sentence(args, test)
    all_text = get_all_text(train, validate, test)
    max_len = len(max(all_text, key=len))
    args.sequence_len = max_len
    all_entity = list(train['entity_token']) + list(
        validate['entity_token']) + list(test['entity_token'])
    max_ent_len = len(max(all_entity, key=len))
    args.ent_len = max_ent_len
    align_data(train, args)
    align_data(validate, args)
    align_data(test, args)
    return train, validate, test


def parse_arguments(parser):
    parser.add_argument('--output_file', type=str, default='Data/twitter/RESULT_text_only', help='')

    parser.add_argument('--dataset', type=str, default='weibo', help='twitter or weibo data')
    parser.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=16, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lambd', type=int, default=1, help='')
    parser.add_argument('--text_only', type=bool, default=False, help='')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')
    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')

    return parser

if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = parse_arguments(parse)
    args = parser.parse_args()

    main(args)
