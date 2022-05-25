import random
import json
import torch
from model import NeuralNet
from nltk_utils import bagOfWords, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = 'data.pth'

data = torch.load(FILE)
inputSize = data['input_size']
hiddenSize = data['hidden_size']
outputSize = data['output_size']
all_words = data['all_words']
tags = data['tags']
modelState = data['model_state']

model = NeuralNet(inputSize, hiddenSize, outputSize).to(device)
model.load_state_dict(modelState)
model.eval()

botName = 'Hailey'
print("Let's chat! type 'quit' to exit")

while True:
    sentence = input('You: ')

    if sentence == 'quit':
        break

    sentence = tokenize(sentence)
    X = bagOfWords(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:

        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{botName}: {random.choice(intent['responses'])}")

    else:
        print(f'{botName}: I do not understand...')