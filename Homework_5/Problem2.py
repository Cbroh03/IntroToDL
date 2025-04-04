import torch
import torch.nn as nn
import torch.optim as optim
from torch import cuda
import requests
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import time

device = 'cuda'

# Pred dataset for training
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text  # This is the entire text data

chars = sorted(list(set(text)))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

# Encode the text into integers
encoded_text = [char_to_int[ch] for ch in text]

class CharDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]

def prepare_dataset(sequence_length):
    sequences = []
    targets = []
    for i in range(0, len(encoded_text) - sequence_length):
        seq = encoded_text[i:i+sequence_length]
        target = encoded_text[i+sequence_length]
        sequences.append(seq)
        targets.append(target)

    # Convert lists to PyTorch tensors
    sequences = torch.tensor(sequences, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    # Instantiate the dataset
    dataset = CharDataset(sequences, targets)

    # Create data loaders
    batch_size = 64
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    return train_loader, test_loader

# Defining the Transformer model
class CharTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead):
        super(CharTransformer, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output[:, -1, :])  # Get the output of the last Transformer block
        return output
    
def training_loop(model, criterion, optimizer, epochs, train_loader, test_loader, max_length):

    train_history = []
    val_history = []
    init_time = time.time()
    print(f"{max_length} sequence transformer results:")

    # Training the model
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        train_history.append(running_loss / len(train_loader))
        val_history.append(correct / total * 100)

        print(f'Epoch {epoch+1}, Loss: {train_history[-1]}, Validation Accuracy: {val_history[-1]}')

    print(f"Training time: {(time.time() - init_time)/60} minutes")
    save_path = f'../../Models/hw5_2_{max_length}.pth'
    #torch.save(model.state_dict(), save_path)

    return train_history, val_history

# Prediction function
def predict_next_char(model, char_to_ix, ix_to_char, initial_str, max_length):
    model.eval()
    with torch.no_grad():
        initial_input = torch.tensor([char_to_ix[c] for c in initial_str[-max_length:]], dtype=torch.long).unsqueeze(0).to(device)
        prediction = model(initial_input)
        predicted_index = torch.argmax(prediction, dim=1).item()
        return ix_to_char[int(predicted_index)]

def next_char_test(model, max_length):
    test_str = "This is a simple example to demonstrate how to predict the next char"
    predicted_char = predict_next_char(model, char_to_int, int_to_char, test_str, max_length)
    print(f"Predicted next character: '{predicted_char}'")

# Hyperparameters
hidden_size = 256
num_layers = 3
nhead = 2

max_lengths = [20, 30, 50]
learning_rate = .0005
epochs = 10

train_histories = []
val_histories = []
models = []

for window_size in max_lengths:
    model = CharTransformer(len(chars), hidden_size, len(chars), num_layers, nhead).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, test_loader = prepare_dataset(window_size)

    train_history, val_history = training_loop(model, criterion, optimizer, epochs, train_loader, test_loader, window_size)
    train_histories.append(train_history)
    val_histories.append(val_history)
    print("\n")

    next_char_test(model, window_size)
    print("\n\n") if window_size != 50 else None
    models.append(model)

    # Plotting training losses
plt.figure(figsize=(10, 5))
for i, max_length in enumerate(max_lengths):
    plt.plot(train_histories[i], label=f"Context Window: {max_length}")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Training Losses over Epochs")
plt.legend()
plt.show()

# Plotting validation accuracies
plt.figure(figsize=(10, 5))
for i, max_length in enumerate(max_lengths):
    plt.plot(val_histories[i], label=f"Context Window: {max_length}")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracies over Epochs")
plt.legend()
plt.show()

###### P2:B
# Hyperparameters
hidden_size = 256
num_layers = 4
nhead = 4

max_lengths = [20, 30, 50]
learning_rate = .0001
epochs = 15

train_histories = []
val_histories = []
models_b = []

for window_size in max_lengths:
    model = CharTransformer(len(chars), hidden_size, len(chars), num_layers, nhead).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, test_loader = prepare_dataset(window_size)

    train_history, val_history = training_loop(model, criterion, optimizer, epochs, train_loader, test_loader, window_size)
    train_histories.append(train_history)
    val_histories.append(val_history)
    print("\n")

    next_char_test(model, window_size)
    print("\n\n") if window_size != 50 else None
    models_b.append(model)

# Plotting training losses
plt.figure(figsize=(10, 5))
for i, max_length in enumerate(max_lengths):
    plt.plot(train_histories[i], label=f"Context Window: {max_length}")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Training Losses over Epochs")
plt.legend()
plt.show()

# Plotting validation accuracies
plt.figure(figsize=(10, 5))
for i, max_length in enumerate(max_lengths):
    plt.plot(val_histories[i], label=f"Context Window: {max_length}")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracies over Epochs")
plt.legend()
plt.show()

