from llava.train.train import train

if __name__ == "__main__":
    import torch
    import random
    seed = 3407
    torch.manual_seed(seed)
    random.seed(seed)
    train(attn_implementation="flash_attention_2")
