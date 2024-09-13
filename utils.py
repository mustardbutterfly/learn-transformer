import torch


def createTargetTokens(inputTokens):
    return {'xs': torch.stack(
        [inputTokensList[:-1] for inputTokensList in inputTokens]
    ), 'ys': torch.stack(
        [inputTokensList[1:] for inputTokensList in inputTokens]
    )}


def generateText(count, tokensContext, embeddingLayer):
    picked = 0
    for _ in range(count):
        embeddedInput = embeddingLayer(torch.tensor([picked], dtype=torch.int64))
        nextTokenLogits = embeddedInput
        normalized = torch.softmax(nextTokenLogits, dim=0)
        picked = torch.multinomial(normalized, 1).item()

        print(tokensContext.getCharFromIdx(picked), end='')
