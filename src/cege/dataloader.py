import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd

class IEMOCAPDataset(Dataset):

    # train = true for training split ids
    # train = false for test split ids
    def __init__(self, train=True):
        # open the pickled features in binary read mode ('rb')
        # using latin1 encoding to handle python2 pickled files in python3
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('./IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

        # label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        
        # create the list of conversation ids the dataset will iterate over
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        # length of the dataset
        self.len = len(self.keys)

    # return a single sample given index
    # each sample is a 7-tuple
    def __getitem__(self, index):
        vid = self.keys[index]

        # uses one hot encoding for speakers
        # male is [1,0]
        # female is [0,1]

        # also returns a list of ones with length equal to the number of utterances in the conversation
        # used as an attention mask

        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    # data is a list of tuples (samples)
    def collate_fn(self, data):
        # convert the list of tuples into a dataframe
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

