import json
from nltk_utils import tokenize, stem, bagOfWords
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignoreWords = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignoreWords]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(tags)

X_train = []
Y_train = []

for (patternSentence, tag) in xy:
    bag = bagOfWords(patternSentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    Y_train.append(label)


X_train = np.array(X_train)
Y_train = np.array(Y_train)


class ChatDataset(Dataset):

    def __init__(self):
        self.nSamples = len(X_train)
        self.xData = X_train
        self.yData = Y_train

    def __getitem__(self, index):
        return self.xData[index], self.yData[index]

    def __len__(self):
        return self.nSamples
inputSize = len(X_train[0])
hiddenSize = 8
outputSize = len(tags)
batch_size = 8
learning_rate = 0.001
numEpochs = 1000

dataset = ChatDataset()
trainLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(inputSize, hiddenSize, outputSize).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(numEpochs):
    for (words, labels) in trainLoader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch [{epoch+1}/{numEpochs}], loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
    'model_state': model.state_dict(),
    'input_size': inputSize,
    'output_size': outputSize,
    'hidden_size': hiddenSize,
    'all_words': all_words,
    'tags': tags
}

FILE = 'data.pth'
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')

