class Vocab:
    def __init__(self):
        self.w2i = {"<blank>": 0}
        self.i2w = {0: "<blank>"}
        self.idx = 1

    def add(self, sentence):
        for ch in sentence.upper():
            if ch not in self.w2i:
                self.w2i[ch] = self.idx
                self.i2w[self.idx] = ch
                self.idx += 1

    def encode(self, sentence):
        return [self.w2i[ch] for ch in sentence.upper() if ch in self.w2i]

    def decode(self, indices):
        return "".join(self.i2w[i] for i in indices if i in self.i2w and i != 0)

    def __len__(self):
        return self.idx
