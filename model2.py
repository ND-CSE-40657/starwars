import torch
import math, collections.abc, random, copy
from layers import *

torch.set_default_device('cpu') # don't use GPU
#torch.set_default_device('cuda') # use GPU

# The maximum length of any sentence, including <BOS> and <EOS>
max_len = 100

def progress(iterable):
    """Iterate over `iterable`, showing progress if appropriate."""
    import os, sys
    if os.isatty(sys.stderr.fileno()):
        try:
            import tqdm
            return tqdm.tqdm(iterable)
        except ImportError:
            return iterable
    else:
        return iterable

class Vocab(collections.abc.MutableSet):
    """Set-like data structure that can change words into numbers and back."""
    def __init__(self):
        words = {'<BOS>', '<EOS>', '<UNK>'}
        self.num_to_word = list(words)    
        self.word_to_num = {word:num for num, word in enumerate(self.num_to_word)}
    def add(self, word):
        if word in self: return
        num = len(self.num_to_word)
        self.num_to_word.append(word)
        self.word_to_num[word] = num
    def discard(self, word):
        raise NotImplementedError()
    def __contains__(self, word):
        return word in self.word_to_num
    def __len__(self):
        return len(self.num_to_word)
    def __iter__(self):
        return iter(self.num_to_word)

    def numberize(self, word):
        """Convert a word into a number."""
        if word in self.word_to_num:
            return self.word_to_num[word]
        else: 
            return self.word_to_num['<UNK>']

    def denumberize(self, num):
        """Convert a number into a word."""
        return self.num_to_word[num]

def read_parallel(filename):
    """Read data from the file named by `filename`.

    The file should be in the format:

    我 不 喜 欢 沙 子 \t I do n't like sand

    where \t is a tab character.

    Argument: filename
    Returns: list of pairs of lists of strings. <BOS> and <EOS> are added to all sentences.
    """
    data = []
    for line in open(filename):
        fline, eline = line.split('\t')
        fwords = ['<BOS>'] + fline.split() + ['<EOS>']
        ewords = ['<BOS>'] + eline.split() + ['<EOS>']
        data.append((fwords, ewords))
    return data

def read_mono(filename):
    """Read sentences from the file named by `filename`.

    Argument: filename
    Returns: list of lists of strings. <BOS> and <EOS> are added to each sentence.
    """
    data = []
    for line in open(filename):
        words = ['<BOS>'] + line.split() + ['<EOS>']
        data.append(words)
    return data
    
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

    def sequence(self, fencs, enums):
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
        
        h = self.decoder.sequence(fencs, ein)
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

if __name__ == "__main__":
    import argparse, sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='training data')
    parser.add_argument('--dev', type=str, help='development data')
    parser.add_argument('infile', nargs='?', type=str, help='test data to translate')
    parser.add_argument('-o', '--outfile', type=str, help='write translations to file')
    parser.add_argument('--load', type=str, help='load model from file')
    parser.add_argument('--save', type=str, help='save model in file')
    args = parser.parse_args()

    if args.train:
        # Read training data and create vocabularies
        traindata = read_parallel(args.train)

        fvocab = Vocab()
        evocab = Vocab()
        for fwords, ewords in traindata:
            fvocab |= fwords
            evocab |= ewords

        # Create model
        m = Model(fvocab, 64, evocab) # try increasing 64 to 128 or 256
        
        if args.dev is None:
            print('error: --dev is required', file=sys.stderr)
            sys.exit()
        devdata = read_parallel(args.dev)
            
    elif args.load:
        if args.save:
            print('error: --save can only be used with --train', file=sys.stderr)
            sys.exit()
        if args.dev:
            print('error: --dev can only be used with --train', file=sys.stderr)
            sys.exit()
        m = torch.load(args.load)

    else:
        print('error: either --train or --load is required', file=sys.stderr)
        sys.exit()

    if args.infile and not args.outfile:
        print('error: -o is required', file=sys.stderr)
        sys.exit()

    if args.train:
        opt = torch.optim.Adam(m.parameters(), lr=0.0003)

        best_dev_loss = None
        for epoch in range(10):
            random.shuffle(traindata)

            ### Update model on train

            train_loss = 0.
            train_ewords = 0
            for fwords, ewords in progress(traindata):
                loss = -m.logprob(fwords, ewords)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
                train_ewords += len(ewords) # includes EOS

            ### Validate on dev set and print out a few translations
            
            dev_loss = 0.
            dev_ewords = 0
            for line_num, (fwords, ewords) in enumerate(devdata):
                dev_loss -= m.logprob(fwords, ewords).item()
                dev_ewords += len(ewords) # includes EOS
                if line_num < 10:
                    translation = m.translate(fwords)
                    print(' '.join(translation))

            if best_dev_loss is None or dev_loss < best_dev_loss:
                best_model = copy.deepcopy(m)
                if args.save:
                    torch.save(m, args.save)
                best_dev_loss = dev_loss

            print(f'[{epoch+1}] train_loss={train_loss} train_ppl={math.exp(train_loss/train_ewords)} dev_ppl={math.exp(dev_loss/dev_ewords)}', flush=True)
            
        m = best_model

    ### Translate test set

    if args.infile:
        with open(args.outfile, 'w') as outfile:
            for fwords in read_mono(args.infile):
                translation = m.translate(fwords)
                print(' '.join(translation), file=outfile)
