import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var
import torch.nn.functional as F

class SamplingReparamKL:
    def __init__(self, latent_dim):
        """Sampling Reparameterization using Optimal KL-Divergence
        between k-dimensional Dirichlet and Logistic Normal.

            mu_i = digamma(alpha_i) - digamma(alpha_k)
            sigma_i = trigammma(alpha_i) - trigamma(alpha_k)

        :param latent_dim: width of the latent dimension
        :param batch_size: batch size for stocastic gradient descent
        :return:
        """
        self.latent_dim = latent_dim

    def to_mu(self, alpha):
        """
        :param alpha: (Tensor)
        :return: (Tensor) mu
        """
        d1 = self.latent_dim - 1
        digamma_d = torch.digamma(alpha[:, -1:])
        mu = torch.digamma(alpha[:, :d1]) - digamma_d
        return mu

    def to_sd(self, alpha):
        """
        :param alpha: (Tensor)
        :return: sigma (Tensor)
        """
        # maybe not have this function.
        # _one = torch.cast(1, dtype=alpha.dtype)
        _one = torch.tensor(1)
        d1 = self.latent_dim - 1
        var = (torch.polygamma( 1 , alpha[:, :d1])
               + torch.polygamma(1 , alpha[:, -1:]))
        sigma = torch.sqrt(var)
        return sigma

    def sample(self, args, batch_size):
        """ Sample from Logistic Normal(mu,sigma) specified by the
        reparameterization

        :param args: (mu, sigma)[Tensor, Tensor]
        :return: z* sample
        """
        mu, sigma = args
        z = to_var(torch.randn([batch_size, self.latent_dim-1]))
        z = mu + sigma * z
        one = torch.ones((batch_size, 1))
        z_star = F.softmax(torch.cat([z, one], dim=1))
        return z_star


class SamplingReparamLaplace:
    def __init__(self, latent_dim):
        """Sampling Reparameterization using Laplace Apprixmation
        between k-dimensional Dirichlet and Logistic Normal.

           mu_i = log(alpha_i) - mean(log(alpha))
           sigma_i = (1 - 2/k) * 1/alpha + 1/k^2 * sum(1/alpha)

        :param latent_dim: width of the latent dimension
        :param batch_size: batch size for stocastic gradient descent
        :return:
        """
        self.latent_dim = latent_dim

    def to_mu(self, alpha):
        """
        :param alpha: (Tensor)
        :return: mu (Tensor)
        """
        log_alpha = torch.log(alpha)
        mean_log_alpha = torch.mean(log_alpha, dim=-1).unsqueeze(-1)
        mu = log_alpha - mean_log_alpha
        return mu

    def to_sd(self, alpha):
        """
        :param alpha: (Tensor)
        :return: sigma (Tensor)
        """
        k1 = 1 - (2 / self.latent_dim)
        k2 = 1 / (self.latent_dim ** 2)
        sigma = k1 * 1/alpha + k2 * torch.sum(1/alpha, dim=-1).unsqueeze(-1)
        return sigma

    def sample(self, args, batch_size):
        """Sample from Logistic Normal(mu,sigma) specified by the
        reparameterization

        :param alpha:
        :return:
        """
        mu, sigma = args
        e = to_var(torch.randn([batch_size, self.latent_dim]))
        z_star = F.softmax(mu + sigma * e)
        return z_star


class VAE_Dir(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        if True:
            self.reparam = SamplingReparamLaplace(latent_size)

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2alpha = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2scale = nn.Linear(hidden_size * self.hidden_factor, latent_size)

        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def forward(self, input_sequence, length):

        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()
            
        ####################################################################
        # REPARAMETERIZATION
        # if we want to change the Dirchlet
        # Modify this part here... 
        # MODIFY (return all the parameter(alpha) and the actual sample)
        alpha = F.tanh(self.hidden2alpha(hidden))
        scale = F.softplus(self.hidden2scale(hidden))
        alpha = torch.exp(scale * alpha)

        mean = self.reparam.to_mu(alpha)
        # maybe reshape to latent_dim

        std = self.reparam.to_sd(alpha)

        z = self.reparam.sample([mean, std], batch_size)

        ####################################################################

        # DECODER
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)


        return logp, alpha, z


    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            ############################# GENERATION OF Z #########################
            beta = torch.ones([batch_size, self.latent_size])
            sum1 = torch.sum(beta, dim=1, keepdim=True)
            alpha = beta/sum1
            mean = self.reparam.to_mu(alpha)
            std = self.reparam.to_sd(alpha)
            z = self.reparam.sample([mean, std], batch_size)
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t=0
        while(t<self.max_sequence_length and len(running_seqs)>0):

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z


    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
