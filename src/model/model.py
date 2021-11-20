import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Function
from transformers import *
from src.utils.utils import to_var
from src.model.attention import Self_Attn, MMAttn, SE_Attn


class ReverseLayerF(Function):
    @staticmethod
    def forward(self, x, args):
        self.lambd = args.lambd
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x):
    return ReverseLayerF.apply(x)



# Neural Network Model (1 hidden layer)
class Twitter_Model(nn.Module):
    def __init__(self, args):
        super(Twitter_Model, self).__init__()
        self.args = args

        vocab_size = args.vocab_size
        emb_dim = args.embed_dim

        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19

        # bert
        bert_model = BertModel.from_pretrained('bert-base-uncased')

        temp_hidden_size = 32

        self.bert_hidden_size = args.bert_hidden_dim
        self.fc2 = nn.Linear(self.bert_hidden_size, temp_hidden_size)

        for param in bert_model.parameters():
            param.requires_grad = False
        self.bertModel = bert_model

        self.dropout = nn.Dropout(args.dropout)

        # IMAGE
        # hidden_size = args.hidden_dim
        vgg_19 = torchvision.models.vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False
        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs,temp_hidden_size)

        # Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(2 * self.hidden_size, 2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        self.mm = SE_Attn(32)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
                to_var(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def cm_attention(self, x ):
        energy = torch.matmul(x, x.permute(1, 0))

        attention = self.softmax(energy)

        out = torch.matmul(attention, x)

        out = self.gamma*out + x

        return out

    def forward(self, text, image, mask, entity):
        """
        Args:
            text (tensor): shape is (bs, 118)
            image (tensor): shape is (bs, 3, 224, 224)
            mask (tensor): shape is (bs, 118)

        Returns:
            [type]: [description]
        """
        # IMAGE
        image = self.vgg(image)  # [bs, 1000]
        image = F.relu(self.image_fc1(image))  # [bs, 32]

        text = self.bertModel(text)[0]  # [bs, 118, 768]
        text = torch.mean(text, dim=1, keepdim=False)  # [bs, 768]
        text = F.relu(self.fc2(text))  # [bs, 32]

        ent = self.bertModel(entity)[0]  # [bs, 17, 768]
        ent = torch.mean(ent, dim=1, keepdim=False) # [bs, 768]
        ent = F.relu(self.fc2(ent)) # [bs, 32]

        attn_text= self.cm_attention(text)
        attn_img = self.cm_attention(image)
        attn_ent = self.cm_attention(ent)

        out = torch.cat([attn_text + attn_ent, attn_img], dim=1) # [bs, 64]

        # out = image

        # text_ent = torch.cat([text , image], dim=1) # [bs, 64]
        # Fake or real
        out = self.class_classifier(out)

        return out 


# Neural Network Model (1 hidden layer)
class Weibo_Model(nn.Module):
    def __init__(self, args):
        super(Weibo_Model, self).__init__()
        self.args = args

        self.event_num = args.event_num

        vocab_size = args.vocab_size
        emb_dim = args.embed_dim

        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19
        temp_hidden_size = 32
        # bert
        bert_model = BertModel.from_pretrained('bert-base-chinese')
        self.bert_hidden_size = args.bert_hidden_dim
        self.fc2 = nn.Linear(self.bert_hidden_size, temp_hidden_size)

        for param in bert_model.parameters():
            param.requires_grad = False
        self.bertModel = bert_model

        self.dropout = nn.Dropout(args.dropout)

        # IMAGE
        vgg_19 = torchvision.models.vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False
        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs, temp_hidden_size)
        

        # Class Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1',
                                         nn.Linear(2 * self.hidden_size, 2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        # attention
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
                to_var(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def cm_attention(self, x ):
        energy = torch.matmul(x, x.permute(1, 0))

        attention = self.softmax(energy)

        out = torch.matmul(attention, x)

        out = self.gamma*out + x

        return out

    def forward(self, text, image, mask, entity):
        # IMAGE
        image = self.vgg(image)  # [N, 1000]
        image = F.relu(self.image_fc1(image))  # (bs, 32)

        text = torch.mean(self.bertModel(text)[0],
                                       dim=1,
                                       keepdim=False)
        text = F.relu(self.fc2(text))  # (bs, 32)

        ent = self.bertModel(entity)[0]  # [bs, 17, 768]
        ent = torch.mean(ent, dim=1, keepdim=False) # [bs, 768]
        ent = F.relu(self.fc2(ent)) # [bs, 32]

        att_img = self.cm_attention(image)
        att_text = self.cm_attention(text)
        att_ent = self.cm_attention(ent)

        out = torch.cat([att_text + att_ent, att_img], dim=1) # [bs, 64]
        # out = text
        # Fake or real
        class_output = self.class_classifier(out)
        return class_output 

