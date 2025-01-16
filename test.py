import torch
import torch.nn.functional as F

import pre_process_data
import models
import train_eval

CONTEXT_SIZE = 5
BATCH_SIZE = 128
learning_rate = 5e-4
MAX_EPOCHS = 300000
MODEL_PATH = "n_gram_weights.pth"

data = pre_process_data.Data("story.txt", expand_factor=10) # i did 10, because the data set overfits due to being small
data.build_dataset_w_split(context_size=CONTEXT_SIZE)

embed_size = int(data.vocab_size ** 0.4)
hidden_size = [100, 100]
model = models.N_Gram(vocab_size=data.vocab_size, context_size=CONTEXT_SIZE, hidden_sizes=hidden_size, embd_size=embed_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.train()

model = train_eval.train_model(model, data.Xtr, data.Ytr, batch_size=BATCH_SIZE, 
                               lr=learning_rate, epochs=MAX_EPOCHS, plot_loss=True, 
                               device=device, save_path_name=MODEL_PATH)

model.eval()

print("Validation")
train_eval.evaluate_split(model, data.Xdev, data.Ydev)

def generate_output(model: torch.nn.Module, itow,context_size, num_of_samples=5, device="cpu"):
    for i in range(num_of_samples):
        context_window = [0]* context_size
        out = []
        model.to(device)
        while True:
            context_tensor = torch.tensor(context_window, device=device).unsqueeze(0)
            logits = model(context_tensor)
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1, replacement=True).item()
            if ix == 0:
                break
            out.append(itow[ix])
            context_window = context_window[1:] + [ix]
        print(" ".join(out))

generate_output(model, data.itow, context_size=CONTEXT_SIZE, num_of_samples=10, device=device)