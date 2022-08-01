import random

def encode(sentence, max_len=512, noise_prob=0.0):
    l = list(sentence.encode('utf_8'))
    l = l + (max_len - len(l)) * [256] # padding
    l = [random.randint(0,255) if random.random() < noise_prob else i for i in l]
    return l[:max_len]

def decode(ids):
    # remove padding
    ids = [i for i in ids if i != 256]
    # decode
    sentence = bytes(ids).decode(encoding='utf-8', errors='ignore')
    return sentence
