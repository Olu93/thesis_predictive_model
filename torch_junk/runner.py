import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.SimpleLSTM import LSTMTagger
from readers import RequestForPaymentLogReader

data = RequestForPaymentLogReader()
vocab_size = data.vocab_len
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, vocab_size)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for instance in data:
        sequence = instance['sequence']
        targets = instance['targets']
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 3. Run our forward pass.
        tag_scores = model()

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()