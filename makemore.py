import mlx.core as mx

learning_rate = 0.01

def one_hot(t, num_classes):
    t_enc = mx.zeros((t.size, num_classes))
    for i,n in enumerate(t):
        t_enc[i, n] = 1
    return t_enc


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
    for w in words[:10000]:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
    
    xs = mx.array(xs)
    num = xs.size
    ys = mx.array(ys)
    print(f"x.shape: {xs.shape}")
    print(f"y.shape: {ys.shape}")
    print(f"num: {num}")
    
    # intialize the network
    print("Initializing network...")
    key = mx.random.key(42)
    W = mx.random.normal(shape=((27,27)), dtype=mx.float32, key=key) # randomly generate 27 neurons' weights. each neuron receives 27 inputs
    X = one_hot(xs, 27).astype(mx.float32) # inputs to neural net
    print(f"X.shape: {X.shape}")
    print(f"W.shape: {W.shape}")
    
    # forward pass
    def loss_fn(W):
        logits = X @ W
        probs = mx.softmax(logits, axis=1)
        print(f"x range: {mx.arange(num)}")
        return -mx.mean(mx.log(probs[mx.arange(num), ys]))
    
    print("Training...")
    # grad_fn = mx.grad(loss_fn)
    # # backward pass
    # grad = grad_fn(W)
    # # update
    # W = W - learning_rate * grad
    # mx.eval(W)
    loss = loss_fn(W)
    print(f"loss: {loss}")


if __name__ == "__main__":
    main()
