import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        #packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []
        #inputs = inputs.unsqueeze(1)
        for i in range(max_len):                                    # maximum sampling length
            hiddens, states = self.lstm(inputs, states)        # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))
            #print(outputs)# (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            #print('predicted',predicted)
            #print(predicted.argmax())
            sampled_ids.append(predicted.tolist()[0])
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                       # (batch_size, 1, embed_size)
        #print('sampled_ids',sampled_ids)
        
"""
    def sample_beam(self, inputs, b=1, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # beam search
        # search to get the best possible sequence using b-size window
        # LogSoftmax is used to avoid vanishing problem since p(sequence) = mul of the probability of each word of the sequence
        # given the previous set of words.
        # It is also equivalent to sum log(p()) hence LogSoftmax is a better option
        # In this implementation of beam search, a b-size window is used at each node that selects the best b possible successor
        # and then select best b overall successor to the current level of nodes.

        # check b for unexpected values
        # perform greedy search by setting b to one
        if b < 1:
            b = 1
        elif b > self.vocab_size:
            b = 1
        # initialize the hidden and cells states to zeros
        (h_init, c_init) = (torch.zeros_like(inputs), torch.zeros_like(inputs))
        # create lists for each of the nodes in the window at every node
        hidden_list = [(h_init, c_init) for _ in range(b)]
        hidden_list_copy = hidden_list.copy()
        input_list = [inputs for _ in range(b)]
        # next_word_candidates variable content is as follows:
        # [(accumulative score, index in the vocab, j to access the hidden lists(hidden list index and other lists)]
        # b*b is the number of words considered to select b words from.
        next_word_candidates = torch.zeros((b*b, 3), device=inputs.device)
        scores_copy = next_word_candidates[:, 0]
        ids_list = torch.tensor([[] for _ in range(b)], device=inputs.device)
        ids_list_copy = ids_list.copy()
        # loop to generate b sentences of length max_len 
        for i in range(max_len):
            for j in range(b):
                # for each selected b words compute the output of the LSTM and the score
                out, hidden_list[j] = self.LSTM_layer(input_list[j], hidden_list[j])
                out = self.fc(out)
                scores = self.Scores(out.squeeze(0).squeeze(0))
                # get b largest scores indices
                max_scores_ids = scores.argsort(descending=True)[0:b]
                # compute and append the new score to form the accumulative score of the sequence 
                next_word_candidates[j*b:j*b+b, 0] = scores[max_scores_ids] + next_word_candidates[j*b:j*b+b, 0]
                # store the new candidates vocab indices 
                next_word_candidates[j*b:j*b+b, 1] = max_scores_ids
                # store the list index
                next_word_candidates[j*b:j*b+b, 2] = torch.tensor([j for _ in range(b)]).reshape((b, 1)).squeeze()
            # In the first step, it is forced to select best possible b words rather than selecting the best possible word from all the 
            # branches b times
            # (In the first step, the initial node is considered as b different nodes and all of the successor nodes of each one of the 
            # initial nodes are the same.)
            if i == 0:
                overall_max_scores_ids = torch.tensor([k for k in range(b)])
            else:
                overall_max_scores_ids = next_word_candidates[:, 0].argsort(dim=0, descending=True)[0:b]
            # save old scores
            scores_copy = next_word_candidates[:, 0]
            # save old hidden list values
            hidden_list_copy = hidden_list.copy()
            # rearranging the data according to the new selected sequences 
            for j in range(b):
                embedding_out = self.embedding_layer(next_word_candidates[overall_max_scores_ids[j], 1].unsqueeze(0).long())
                input_list[j] = embedding_out.unsqueeze(1)
                hidden_list[j] = hidden_list_copy[int(next_word_candidates[overall_max_scores_ids[j], 2])]
                ids_list[j] = list(ids_list_copy[int(next_word_candidates[overall_max_scores_ids[j], 2])])
                ids_list[j].append(int(next_word_candidates[overall_max_scores_ids[j], 1]))
                next_word_candidates[j*b:j*b+b, 0] = torch.tensor([scores_copy[overall_max_scores_ids[j]] for _ in range(b)])
            # save new words sequence 
            ids_list_copy = ids_list.copy()
        # select the best sentence 
        ids_list_winner = ids_list[0]
        # print("all:\n", ids_list)
        # print('\n')
        return ids_list_winner
     """        
        

        #print(sampled_ids.squeeze())
        #sampled_ids = torch.cat(sampled_ids, 1)                # (batch_size, 20)
        return sampled_ids#.squeeze()
