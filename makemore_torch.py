import torch
import torch.nn.functional as F

def main():
    words = open('names.txt', 'r').read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi["."] = 0
    itos = {i+1:s for i,s in enumerate(chars)}
    itos[0] = "."

    # create a training set of bigrams
    print("Creating training set...")
    xs, ys = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
    
    xs = torch.tensor(xs)
    num = xs.nelement()
    ys = torch.tensor(ys)
    print("xs.shape", xs.shape)
    print("ys.shape", ys.shape)
    print("num", num)

    g = torch.Generator().manual_seed(42)
    W = torch.randn((27, 27), generator=g, requires_grad=True)
    print("W.shape", W.shape)
    x_enc = F.one_hot(xs, num_classes=27).float()
    print("x_enc.shape", x_enc.shape)
    logits = x_enc @ W
    print("logits.shape", logits.shape)
    counts = logits.exp()
    print("counts.shape", counts.shape)
    probs = counts / counts.sum(1, keepdim=True)
    print("probs.shape", probs.shape)
    loss = -probs[torch.arange(num), ys].log().mean()
    print("loss", loss.item())
    W.grad = None
    loss.backward()
    print(loss)

if __name__ == "__main__":
    main()
