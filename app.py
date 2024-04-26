from flask import Flask, render_template, request
import torch
import torch.nn as nn
import spacy
import pandas as pd

app = Flask(__name__)

# Load word_to_index mapping from file
word_to_index = {}
with open("word_to_index.txt", "r") as file:
    for line in file:
        word, index = line.strip().split(": ")
        word_to_index[word] = int(index)

# Load vocab from file
vocab = []
with open("vocab_data", "r") as file:
    for line in file:
        vocab.append(line.strip())

df = pd.read_csv('review.csv')

# Define your LSTMModel class here
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = len(vocab) + 1  # Add 1 for padding token
embedding_dim = 128
hidden_dim = 256
output_dim = 1
n_layers = 2
dropout = 0.5
batch_size = 64
learning_rate = 0.001
num_epochs = 3

# Load the pre-trained model
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout).to(device)
model.load_state_dict(torch.load('best-model.pt'))
model.eval()

# Load Spacy for tokenization
nlp = spacy.load("en_core_web_sm")

# Define function for tokenizing user input
def tokenize_input(review):
    tokens = [token.text for token in nlp(review.lower()) if not token.is_stop and not token.is_punct]
    return tokens

max_len = max(df["numericalized_review"].apply(len))
def pad_sequence(sequence, max_len):
    return sequence + [0]*(max_len - len(sequence))

# Define route for the home page
@app.route('/')
def home():
    return render_template('result.html')

# Define route for handling prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        tokenized_review = tokenize_input(review)
        numericalized_review = [word_to_index[token] if token in word_to_index else 0 for token in tokenized_review]
        padded_review = pad_sequence(numericalized_review, max_len)
        tensor_review = torch.tensor(padded_review).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = torch.sigmoid(model(tensor_review)).item()
        sentiment = "positive" if prediction >= 0.5 else "negative"
        return render_template('result.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
