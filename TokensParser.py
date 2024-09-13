import torch

from utils import createTargetTokens

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TokensParser:
    def __init__(self, trainPercent):
        torch.manual_seed(342299535)

        inputText = open('input.txt', 'r').read()
        chars = set(inputText)
        chars = sorted(chars)

        self._mapOfCharsToIdx = {c: index for index, c in enumerate(chars)}
        self._mapOfIdxToChars = {index: c for index, c in enumerate(chars)}
        tokenizedText = torch.tensor([self._mapOfCharsToIdx[x] for x in inputText]).to(device)

        trainIndex = int(len(tokenizedText) * trainPercent)

        self._tokenizedTextTrain = tokenizedText[:trainIndex]
        self._tokenizedTextTest = tokenizedText[trainIndex:]

    def getTokenizedDataBatch(self, train=True, contextCount=8, batchSize=4):
        xs = []
        for _ in range(batchSize):
            if train:
                startIndex = torch.randint(0, len(self._tokenizedTextTrain) - contextCount, (1,))[0].item()
                xs.append(self._tokenizedTextTrain[startIndex: startIndex + contextCount + 1])
            else:
                startIndex = torch.randint(0, len(self._tokenizedTextTest) - contextCount, (1,))[0].item()
                xs.append(self._tokenizedTextTest[startIndex: startIndex + contextCount + 1])
        return createTargetTokens(xs)

    def getTestData(self):
        return {'xs': self._tokenizedTextTest[:-1], 'ys': self._tokenizedTextTest[1:]}

    def getCharFromIdx(self, idx):
        return self._mapOfIdxToChars[idx]
