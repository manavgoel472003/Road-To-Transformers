import torch
import torch.nn.functional as F

import pre_process_data
import models
import train_eval

CONTEXT_SIZE = 5 # as we reduce it
BATCH_SIZE = 64
learning_rate = 1e-4
MAX_EPOCHS = 400000
MODEL_PATH = "stored_weights/lstm.pth"

data = pre_process_data.Data("story.txt", expand_factor=10) # i did 10, because the data set overfits due to being small
data.build_dataset_w_split(context_size=CONTEXT_SIZE)

embed_size = int(data.vocab_size ** 0.4)
hidden_size = 150

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.LSTM(embed_size=embed_size, hidden_size=hidden_size, vocab_size=data.vocab_size, device=device)
model.train()

X, Y = data.Xtr[:,:-1], data.Xtr[:,1:]
model = train_eval.train_model(model, X, Y, seq=True, batch_size=BATCH_SIZE, 
                               lr=learning_rate, epochs=MAX_EPOCHS, plot_loss=True, 
                               device=device, save_path_name=MODEL_PATH)

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print("Validation")
x_ev, y_ev = data.Xdev[:1000, :-1], data.Xdev[:1000, 1:]
train_eval.evaluate_split(model, x_ev, y_ev, seq=True, device=device)


def generate_output(model: torch.nn.Module, itow, num_of_samples=5, max_sen_len=16, device="cpu"):
    for _ in range(num_of_samples):
        inputs = [0]
        generated_tokens = []
        count = 0
        while True and count < max_sen_len:
            count += 1
            input_tensor = torch.tensor(inputs, dtype=torch.long, device=device).unsqueeze(0)
            out, _ = model(input_tensor)
            probs = F.softmax(out[-1], dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            if next_token == 0:
                break
            generated_tokens.append(itow[next_token])
            inputs.append(next_token)
        print(" ".join(generated_tokens))

generate_output(model, data.itow, num_of_samples=10, device=device)

print("Test")
x_ev, y_ev = data.Xte[:100, :-1], data.Xte[:100, 1:]
train_eval.evaluate_split(model, x_ev, y_ev, seq=True, device=device)