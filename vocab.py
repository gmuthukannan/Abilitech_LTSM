class Vocab:
    def __init__(self):
        # Index 0: CTC blank  1: SOS  2: EOS  3+: characters
        self.w2i = {"<blank>": 0, "<sos>": 1, "<eos>": 2}
        self.i2w = {0: "<blank>", 1: "<sos>", 2: "<eos>"}
        self.idx = 3

    @property
    def sos_id(self): return 1

    @property
    def eos_id(self): return 2

    def add(self, sentence):
        for ch in sentence.upper():
            if ch not in self.w2i:
                self.w2i[ch] = self.idx
                self.i2w[self.idx] = ch
                self.idx += 1

    def encode(self, sentence):
        return [self.w2i[ch] for ch in sentence.upper() if ch in self.w2i]

    def decode(self, indices):
        skip = {0, 1, 2}  # blank, sos, eos
        return "".join(self.i2w[i] for i in indices if i in self.i2w and i not in skip)

    def __len__(self):
        return self.idx
