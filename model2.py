import torch
import math, random, copy, sys, os

# Directories in GitHub repository
data_dir = './data'
lib_dir = '.'
out_dir = '.'

# Directories on Kaggle
#data_dir = '/kaggle/input/star-wars-chinese-english/data'
#lib_dir = '/kaggle/input/star-wars-chinese-english'
#out_dir = '/kaggle/working'

sys.path.append(lib_dir)
from layers import *
from utils import *
import bleu

# Which training set to use
train_name = 'small'
#train_name = 'large'

torch.set_default_device('cpu') # don't use GPU
#torch.set_default_device('cuda') # use GPU

# The maximum length of any sentence, including <BOS> and <EOS>
max_len = 256

# The original Model 2 had two tables t(e|f) and a(j|i). Here, we
# factor t(e|f) into two matrices (called U and V in the notes), and
# a(j|i) into two matrices M and Nᵀ. This makes the whole model break
# into two parts, an encoder (V and N) and a decoder (U and M). V[f]
# can be thought of as a vector representation of f, and U[:,e] can be
# thought of as a vector representation of e. Likewise, N[j] can be
# thought of as a vector representation of j, and M[i] can be thought
# of as a vector representation of i.
    
class Encoder(torch.nn.Module):
    """IBM Model 2 encoder."""

    def __init__(self, vocab_size, dims):
        super().__init__()
        self.emb = Embedding(vocab_size, dims) # This is called V in the notes
        self.pos = Embedding(max_len, dims) # N

    def forward(self, fnums):
        """Encode a Chinese sentence.

        Argument: Chinese sentence (list of n ints)
        Returns: Chinese word encodings (Tensor of size n,2d)"""

        # Pack femb (word embeddings) and fpos (position embeddings) into single vector
        femb = self.emb(fnums)
        fpos = self.pos(torch.arange(len(fnums)))
        return torch.cat([femb, fpos], dim=-1)

class Decoder(torch.nn.Module):
    """IBM Model 2 decoder."""
    
    def __init__(self, dims, vocab_size):
        super().__init__()
        self.dims = dims
        self.pos = Embedding(max_len, dims) # M
        self.out = SoftmaxLayer(dims, vocab_size) # This is called U in the notes

    def start(self, fencs):
        """Return the initial state of the decoder.

        Argument:
        - fencs (Tensor of size n,2d): Source encodings

        For Model 2, the state is just the English position, but in
        general it could be anything. If you add an RNN to the
        decoder, you should call the RNN's start() method here.
        """
        
        return (fencs, 0)

    def step(self, state, enum):
        """Input an English word (enum) and output the log-probability
        distribution over the next English word.

        Arguments:
            state: Old state of decoder
            enum:  Next English word (int)

        Returns: (state, out), where
            state: New state of decoder
            out:   Vector of log-probabilities (Tensor of size len(evocab))

        """
        
        (fencs, i) = state

        # Unpack fencs into fembs (word embeddings) and fpos (position embeddings)
        d = self.dims
        fembs = fencs[...,:d]           # n,d
        fpos = fencs[...,d:]            # n,d

        # Compute t(e | f_j) for all j
        v = self.out(fembs)             # n,len(evocab)

        # Compute queries and keys based purely on positions
        q = self.pos(i)                 # d
        k = fpos                        # n,d

        # Compute expected output
        o = attention(q, k, v)          # len(evocab)
        
        return ((fencs, i+1), o)

    def forward(self, fencs, enums):
        """Compute probability distributions for an English sentence.

        Arguments:
            fencs: Chinese word encodings (Tensor of size n,2d)
            enums: English words, including <BOS> but not <EOS> (list of m ints)

        Returns: Matrix of log-probabilities (Tensor of size m,len(evocab))
        """
        d = self.dims
        fembs = fencs[...,:d]           # n,d
        fpos = fencs[...,d:]            # n,d
        m = len(enums)
        v = self.out(fembs)             # n,len(evocab)
        q = self.pos(torch.arange(m))   # m,d
        k = fpos                        # n,d
        o = attention(q, k, v)          # m,len(evocab)
        return o

class Model(torch.nn.Module):
    """IBM Model 2.

    You are free to modify this class, but you probably don't need to;
    it's probably enough to modify Encoder and Decoder.
    """
    def __init__(self, fvocab, dims, evocab):
        super().__init__()

        # Store the vocabularies inside the Model object
        # so that they get loaded and saved with it.
        self.fvocab = fvocab
        self.evocab = evocab
        
        self.encoder = Encoder(len(fvocab), dims)
        self.decoder = Decoder(dims, len(evocab))

    def logprob(self, fwords, ewords):
        """Return the log-probability of a sentence pair.

        Arguments:
            fwords: source sentence (list of str)
            ewords: target sentence (list of str)

        Return:
            log-probability of ewords given fwords (scalar)"""

        fnums = torch.tensor([self.fvocab.numberize(f) for f in fwords])
        fencs = self.encoder(fnums)
        
        enums = torch.tensor([self.evocab.numberize(e) for e in ewords])
        ein = enums[:-1] # no <EOS>
        eout = enums[1:] # no <BOS>
        
        h = self.decoder(fencs, ein)
        logprobs = h[torch.arange(len(eout)), eout] # logprobs[i] = h[i,eout[i]]
        return logprobs.sum()

    def translate(self, fwords):
        """Translate a sentence using greedy search.

        Arguments:
            fwords: source sentence (list of str)

        Return:
            ewords: target sentence (list of str)
        """
        
        fnums = torch.tensor([self.fvocab.numberize(f) for f in fwords])
        fencs = self.encoder(fnums)
        state = self.decoder.start(fencs)
        ewords = []
        enum = self.evocab.numberize('<BOS>')
        for i in range(max_len-1):
            (state, elogprobs) = self.decoder.step(state, enum)
            enum = torch.argmax(elogprobs).item()
            eword = self.evocab.denumberize(enum)
            if eword == '<EOS>': break
            ewords.append(eword)
        return ewords

def train(train_data, dev_data):
    fvocab = Vocab()
    evocab = Vocab()
    for fwords, ewords in train_data:
        fvocab |= fwords
        evocab |= ewords

    model = Model(fvocab, 64, evocab) # try other values
    
    opt = torch.optim.Adam(model.parameters(), lr=0.0003)

    best_dev_bleu = None
    for epoch in range(2):
        random.shuffle(train_data)

        ### Update model on train

        train_loss = 0.
        train_ewords = 0
        for fwords, ewords in progress(train_data):
            loss = -model.logprob(fwords, ewords)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            train_ewords += len(ewords)-1 # includes EOS but not BOS

        ### Validate on dev set and print out a few translations

        dev_loss = 0.
        dev_ewords = 0
        dev_outputs = []
        for line_num, (fwords, ewords) in enumerate(dev_data):
            dev_loss -= model.logprob(fwords, ewords).item()
            dev_ewords += len(ewords)-1 # includes EOS but not BOS

            output = model.translate(fwords)
            dev_outputs.append(output)
            if line_num < 10:
                print(' '.join(output), file=sys.stderr, flush=True)

        dev_refs = [ewords for (_, ewords) in dev_data]
        dev_bleu = bleu.score(dev_outputs, dev_refs)
        if best_dev_bleu is None or dev_bleu > best_dev_bleu:
            best_model = copy.deepcopy(model)
            best_dev_bleu = dev_bleu

        print(f'[{epoch+1}] train_loss={train_loss} train_ppl={math.exp(train_loss/train_ewords)} dev_ppl={math.exp(dev_loss/dev_ewords)} dev_bleu={dev_bleu}', file=sys.stderr, flush=True)

    return best_model

if __name__ == "__main__":
    train_data = read_parallel(os.path.join(data_dir, f'{train_name}.zh'),
                              os.path.join(data_dir, f'{train_name}.en'))
    dev_data = read_parallel(os.path.join(data_dir, 'dev.zh'),
                            os.path.join(data_dir, 'dev.reference.en'))
    model = train(train_data, dev_data)
    
    #model = torch.load(os.path.join(out_dir, 'mymodel.pt'))
    torch.save(model, os.path.join(out_dir, 'mymodel.pt'))

    test_inputs = read_mono(os.path.join(data_dir, 'test.zh'))
    test_outputs = [model.translate(fwords) for fwords in test_inputs]
    test_refs = read_mono(os.path.join(data_dir, 'test.reference.en'))
    write_mono(test_outputs, os.path.join(out_dir, 'test.model2.en'))

    print(f'[done] test_bleu={bleu.score(test_outputs, test_refs)}')
