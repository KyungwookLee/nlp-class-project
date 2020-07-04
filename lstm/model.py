import random

import torch 
import torch.nn as nn

from attention import AttentionBase, DotAttention, ConcatAttention
from dataset import Language, NmtDataset, collate_fn

class Seq2Seq(torch.nn.Module):
    def __init__(self, src: Language, trg: Language, attention_type: str, embedding_dim: int=128, hidden_dim: int=64):
        """ Seq2Seq with Attention model

        Parameters:
        src -- source language vocabs
        trg -- target language vocabs
        attention_type -- internal attention type: 'dot' or 'concat'
        embeding_dim -- embedding dimension
        hidden_dim -- hidden dimension
        """
        super().__init__()
        PAD = Language.PAD_TOKEN_IDX
        SRC_TOKEN_NUM = len(src.idx2word) # Including <PAD>
        TRG_TOKEN_NUM = len(trg.idx2word) # special tokens + 8 tokens (E, I, S, N, T, F, P, J)

        ### Declare Embedding Layers
        # Doc for Embedding Layer: https://pytorch.org/docs/stable/nn.html#embedding 
        #
        # Note: You should set padding_idx options to embed <PAD> tokens to 0 values 
        self.src_embedding: nn.Embedding = nn.Embedding(SRC_TOKEN_NUM, embedding_dim, padding_idx=PAD)
        self.trg_embedding: nn.Embedding = nn.Embedding(TRG_TOKEN_NUM, embedding_dim, padding_idx=PAD)

        ### Declare LSTM/LSTMCell Layers
        # Doc for LSTM Layer: https://pytorch.org/docs/stable/nn.html#lstm
        # Doc for LSTMCell Layer: https://pytorch.org/docs/stable/nn.html#lstmcell
        # Explanation for bidirection RNN in torch by @ceshine_en (English): https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
        #
        # Note 1: Don't forget setting batch_first option because our tensor follows [batch_size, sequence_length, ...] form.
        # Note 2: Use one layer LSTM with bias for encoder & decoder
        self.encoder: nn.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bias=True, batch_first=True, dropout=0.3, bidirectional=True)  # 2*hidden?
        self.decoder: nn.LSTMCell = nn.LSTMCell(embedding_dim, 2 * hidden_dim, bias=True)  # 2*hidden?

        # Attention Layer
        if attention_type == 'dot':
            self.attention: AttentionBase = DotAttention()
        elif attention_type == 'concat':
            self.attention: AttentionBase = ConcatAttention(hidden_dim)

        ### Declare output Layers
        # Output layers to attain combined-output vector o_t in the handout
        # Doc for Sequential Layer: https://pytorch.org/docs/stable/nn.html#sequential
        # Doc for Linear Layer: https://pytorch.org/docs/stable/nn.html#linear 
        # Doc for Tanh Layer: https://pytorch.org/docs/stable/nn.html#tanh 
        #
        # Note: Shape of combined-output o_t should be (batch_size, TRG_TOKEN_NUM - 1) because we will exclude the probability of <PAD>
        self.output: nn.Sequential = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, TRG_TOKEN_NUM - 4, bias=False)
            #nn.Linear(hidden_dim, 2, bias=False)    # 2 output type ((E, I), (S, N), (T, F), (P, J))
        )

        ### END YOUR CODE

    def forward(self, src_sentences: torch.Tensor, trg_sentences: torch.Tensor, teacher_force: float=.5):
        """ Seq2Seq forward function
        Note: As same with assignment 1, <PAD> should not affect the loss calculation.

        Parameters:
        src_sentences -- batched source sentences
                            in shape (batch_size, sentence_length)
        trg_sentences -- batched target sentences
                            in shape (batch_size, sentence_length)
        teacher_force -- the probability of teacher forcing

        Return:
        loss -- average loss per a non-<PAD> word
        """
        # You may use below notations
        batch_size = src_sentences.shape[0]
        PAD = Language.PAD_TOKEN_IDX
        SOS = Language.SOS_TOKEN_IDX
        EOS = Language.EOS_TOKEN_IDX

        encoder_masks = src_sentences == PAD

        ### Encoder part (~7 lines)
        # We strongly recommand you to use torch.nn.utils.rnn.pack_padded_sequence/pad_packed_sequence to deal with <PAD> and boost the performance.
        # Doc for pack_padded_sequence: https://pytorch.org/docs/stable/nn.html#pack-padded-sequence 
        # Doc for pad_packed_sequence: https://pytorch.org/docs/stable/nn.html#pad-packed-sequence
        # Explanation for PackedSequence by @simonjisu (Korean): https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html
        # Because you have already sorted sentences at collate_fn, you can use pack_padded_sequence without any modification.
        #
        # Variable:
        # encoder_hidden -- encoder_hidden is encoder hidden state which is same with h^enc in the handout 
        #                   in shape (batch_size, sequence_length, hidden_dim * 2)
        #                   All values in last dimension (hidden_dim dimension) are zeros for <PAD> location.
        # hidden_state -- Last encoder hidden state
        #                   in shape (batch_size, hidden_dim * 2)
        # cell_state -- Last encoder cell state
        #                   in shape (batch_size, hidden_dim * 2)
        src_input = self.src_embedding(src_sentences)
        src_packed_lengths = torch.sum(encoder_masks == 0, dim=1)
        packed_input = nn.utils.rnn.pack_padded_sequence(src_input, src_packed_lengths, batch_first=True)
        packed_output, (encoder_hidden_t, encoder_cell_t) = self.encoder(packed_input)
        encoder_hidden, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, padding_value=PAD)
        hidden_state = torch.cat((encoder_hidden_t[0, :, :], encoder_hidden_t[1, :, :]), dim=1)
        cell_state = torch.cat((encoder_cell_t[0, :, :], encoder_cell_t[1, :, :]), dim=1)

        ### END YOUR CODE

        # Loss initialize
        loss = encoder_hidden.new_zeros(())

        outputs = torch.empty([batch_size])
        decoder_out = trg_sentences.new_full([batch_size], fill_value=SOS)
        for trg_word_idx in range(trg_sentences.shape[1] - 2):
            # Teacher forcing: feed correct labels with a probability of teacher_force
            decoder_input = trg_sentences[:, trg_word_idx] if torch.distributions.bernoulli.Bernoulli(teacher_force).sample() else decoder_out

            # You may use below notations
            decoder_target = trg_sentences[:, trg_word_idx+1]
            decoder_mask = trg_sentences[:, trg_word_idx+1] == PAD

            ### Decoder part (~8 lines)
            # Calculate loss for each word in a batch
            #
            # Note 1: the probability of <PAD> should be zero as we did in Assignment 1. 
            # Because combined-output o_t is already in shape (batch_size, TRG_TOKEN_NUM - 1), this output becomes the probability of target tokens except <PAD>. 
            # 
            # Note 2: You should exclude losses by <PAD> targets. ignore_index argument would be useful.
            #
            # Note 3: in fact, in most cases, you don't need to strictly exclude <PAD> tokens like this assignment.
            # You can train the model properly even if <PAD> probability is not zeros. <PAD> takes very tiny proportion in loss calculation.
            # However, we have already stritctly excluded in assignment 1, so let's just keep the strict policy.
            #
            # Variable:
            # loss -- loss sum of non-<PAD> target words
            #           in shape () - scalar tensor
            #           You should aggregate whole losses of non-<PAD> tokens in a batch
            # decoder_out -- word indices which have largest probability for next words
            #           in shape (batch_size, )
            trg_input = self.trg_embedding(decoder_input)
            decoder_hidden_t, decoder_cell_t = self.decoder(trg_input, (hidden_state, cell_state)) if trg_word_idx == 0 else self.decoder(trg_input, (decoder_hidden_t, decoder_cell_t))
            decoder_hidden_t[decoder_mask, :] = 0
            attention_t, distribution_t = self.attention(encoder_hidden, encoder_masks, decoder_hidden_t, decoder_mask)
            output_t = self.output(torch.cat((decoder_hidden_t, attention_t), dim=1))
            prob_output = nn.functional.softmax(output_t, dim=1)
            decoder_out = prob_output.max(dim=1)[1] + 4    #1
            loss += nn.functional.cross_entropy(output_t, decoder_target-4, ignore_index=-4, reduction='sum')
            #decoder_out = prob_output.max(dim=1)[1]
            #loss += nn.functional.cross_entropy(output_t, decoder_target, reduction='sum')
            outputs = decoder_out[:,None] if trg_word_idx==0 else torch.cat((outputs, decoder_out[:,None]), 1)

            ### END YOUR CODE
        
        assert loss.shape == torch.Size([])
        return loss / (trg_sentences[:, 1:-1] != PAD).sum(), outputs # Return average loss per a non-<PAD> word


    def predict(self, sentence: torch.Tensor, split=False, max_len: int=30):
        """
        Parameters:
        sentence -- sentence to be translated
                        in shape (sentence_length, )
        max_len -- maximum word length of the translated stentence

        Return:
        translated -- translated sentence
                        in shape (translated_length, ) of which translated_length <= max_len
                        with torch.long type
        distrubutions -- stacked attention distribution
                        in shape (translated_length, sentence_length)
                        This is used for ploting
        """
        PAD = Language.PAD_TOKEN_IDX
        SOS = Language.SOS_TOKEN_IDX
        EOS = Language.EOS_TOKEN_IDX
        sentence_length = sentence.size(0)

        ### YOUR CODE HERE (~ 22 lines)
        # You can complete this part without any help by refering to the above forward function
        # Note: use argmax to get the next input word
        src_hidden = self.src_embedding(sentence)
        encoder_hidden, (encoder_hidden_t, encoder_cell_t) = self.encoder(src_hidden.unsqueeze(0))
        hidden_state = torch.cat((encoder_hidden_t[0, :, :], encoder_hidden_t[1, :, :]), dim=1)
        cell_state = torch.cat((encoder_cell_t[0, :, :], encoder_cell_t[1, :, :]), dim=1)
        encoder_mask = sentence.unsqueeze(0) == PAD

        pred_list = []
        attention_list = []
        decoder_out = sentence.new_full([1], fill_value=SOS)
        iters = 4 if split else 1
        for i in range(iters):
        #while decoder_out != EOS and len(translated_list) < max_len:
            decoder_input = self.trg_embedding(decoder_out)

            decoder_hidden_t, decoder_cell_t = self.decoder(decoder_input, (hidden_state, cell_state)) if i == 0 else self.decoder(decoder_input, (decoder_hidden_t, decoder_cell_t))
            decoder_mask = decoder_out == PAD
            decoder_hidden_t[decoder_mask] = 0
            attention_t, distribution_t = self.attention(encoder_hidden, encoder_mask, decoder_hidden_t, decoder_mask)
            output_t = self.output(torch.cat((decoder_hidden_t, attention_t), dim=1))
            prob_output = nn.functional.softmax(output_t, dim=1)
            decoder_out = prob_output.max(dim=1)[1] + 4
            pred_list.extend(decoder_out)
            attention_list.append(distribution_t[0,:].tolist())
            #attention_list = torch.cat((attention_list, distribution_t), dim=1)

        pred = torch.LongTensor(pred_list)
        attention = torch.Tensor(attention_list)
        ### END YOUR CODE

        assert pred.dim() == 1 and attention.shape == torch.Size([pred.size(0), sentence_length])
        return pred, attention

#############################################
# Testing functions below.                  #
#############################################

def test_initializer_and_forward():
    print("=====Model Initializer & Forward Test Case======")
    french = Language(path='data/train.fr.txt')
    english = Language(path='data/train.en.txt')
    french.build_vocab()
    english.build_vocab()
    dataset = NmtDataset(src=french, trg=english)

    model = Seq2Seq(french, english, attention_type='dot')

    # the first test
    try:
        model.load_state_dict(torch.load("sanity_check.pth", map_location='cpu'))
    except Exception as e:
        print("Your model initializer is wrong. Check the handout and comments in details and implement the model precisely.")
        raise e
    print("The first test passed!")

    batch_size = 8
    max_pad_len = 5
    sentence_length = list(map(lambda pair: (len(pair[0]), len(pair[1])), dataset))

    batch_indices = [[0, 1, 2, 3, 4, 5, 6, 7]]
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, collate_fn=collate_fn, num_workers=0, batch_sampler=batch_indices)
    batch = next(iter(dataloader))
    loss = model(batch[0], batch[1])

    # the second test
    assert loss.detach().allclose(torch.tensor(3.03703070), atol=1e-7), \
        "Loss of the model does not match expected result."
    print("The second test passed!")

    loss.backward()

    # the third test
    expected_grad = torch.Tensor([[-8.29117271e-05, -4.44278521e-05, -2.64967621e-05],
                                  [-3.89243884e-04, -1.29778590e-03, -4.56827343e-04],
                                  [-2.76966626e-03, -1.00148167e-03, -6.68873254e-05]])
    assert model.encoder.weight_ih_l0.grad[:3,:3].allclose(expected_grad, atol=1e-7), \
        "Gradient of the model does not match expected result."
    print("The third test passed!")

    print("All 3 tests passed!")

def test_translation():
    print("=====Translation Test Case======")
    french = Language(path='data/train.fr.txt')
    english = Language(path='data/train.en.txt')
    french.build_vocab()
    english.build_vocab()

    model = Seq2Seq(french, english, attention_type='dot')
    model.load_state_dict(torch.load("sanity_check.pth", map_location='cpu'))

    sentence = torch.Tensor([ 4,  6, 40, 41, 42, 43, 44, 13]).to(torch.long)
    translated, distributions = model.translate(sentence)

    # the first test
    assert translated.tolist() == [4, 16, 9, 56, 114, 51, 1, 14, 3], \
        "Your translation does not math expected result."
    print("The first test passed!")

    # the second test
    expected_dist = torch.Tensor([[9.98170257e-01, 1.74237683e-03, 7.48323873e-05],
                                  [1.94309454e-03, 9.82858062e-01, 4.87918453e-03],
                                  [2.26807110e-02, 7.29433298e-02, 3.17393959e-01]])
    assert distributions[:3,:3].allclose(expected_dist, atol=1e-7), \
        "Your attetion distribution does not math expected result."
    print("The second test passed!")

    # the third test
    sentence = torch.Tensor([ 4,  6, 40, 41, 42, 43, 44, 13]).to(torch.long)
    translated, _ = model.translate(sentence, max_len=4)
    assert translated.tolist() == [4, 16, 9, 56], \
        "max_len parameter dose not work properly."
    print("The third test passed!")

    print("All 3 tests passed!")

if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    random.seed(1234)
    torch.manual_seed(1234)

    test_initializer_and_forward()
    test_translation()