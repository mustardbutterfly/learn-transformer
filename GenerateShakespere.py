import torch

from BigramModel import BigramModel
from TokensParser import TokensParser
from utils import generateText

device = 'cuda' if torch.cuda.is_available() else 'cpu'
EmbeddingSize = 64
numberOfChars = 65

tokensContext = TokensParser(.9)

model = BigramModel(EmbeddingSize)
model.to(device)
embeddingLayer = torch.nn.Embedding(numberOfChars, EmbeddingSize)
embeddingLayer.to(device)
outputLayer = torch.nn.Linear(EmbeddingSize, numberOfChars)
outputLayer.to(device)
softmax = torch.nn.Softmax(dim=0)
softmax.to(device)

for i in range(200000):
    trainingDataBatch = tokensContext.getTokenizedDataBatch()
    xs = trainingDataBatch['xs']
    embeddedInput = embeddingLayer(xs)  # 4 x 8 x 64
    trainedInput = model(embeddedInput)
    yPred = softmax(outputLayer(trainedInput))  # remaining layers from the paper
    loss = torch.nn.functional.cross_entropy(yPred.view(-1, numberOfChars), trainingDataBatch['ys'].view(-1))
    if i % 1000 == 0:
        print(loss.item())
    model.zero_grad()
    loss.backward()
    torch.optim.AdamW(model.parameters())

testDataBatch = tokensContext.getTokenizedDataBatch(train=False)
xs = testDataBatch['xs']
embeddedInput = embeddingLayer(xs)
trainedInput = model(embeddedInput)
yPred = softmax(outputLayer(trainedInput))  # remaining layers from the paper
loss = torch.nn.functional.cross_entropy(yPred.view(-1, numberOfChars), testDataBatch['ys'].view(-1))
print(loss.item())

generateText(100, tokensContext, embeddingLayer)
