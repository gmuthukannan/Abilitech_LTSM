class Vocab:
    def __init__(self):
        self.w2i = {"<blank>": 0}  # CTC blank must be index 0
        self.i2w = {0: "<blank>"}
        self.idx = 1

    def add(self, sentence):
        for w in sentence.upper().split():
            if w not in self.w2i:
                self.w2i[w] = self.idx
                self.i2w[self.idx] = w
                self.idx += 1

    def encode(self, sentence):
        return [self.w2i[w] for w in sentence.upper().split() if w in self.w2i]

    def decode(self, indices):
        return " ".join(self.i2w[i] for i in indices if i in self.i2w and i != 0)

    def __len__(self):
        return self.idx
