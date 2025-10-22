# %% [markdown]
# # LSTM-arithmetic
#
# ## Dataset
# - [Arithmetic dataset](https://drive.google.com/file/d/1cMuL3hF9jefka9RyF4gEBIGGeFGZYHE-/view?usp=sharing)

# %%
# ! pip install seaborn
# ! pip install opencc
# ! pip install -U scikit-learn

from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.utils.rnn
import torch.utils.data
import matplotlib.pyplot as plt
import seaborn as sns
import opencc
import os
from sklearn.model_selection import train_test_split

data_path = './data'

# %%
df_train = pd.read_csv(os.path.join(data_path, 'arithmetic_train.csv'))
df_eval = pd.read_csv(os.path.join(data_path, 'arithmetic_eval.csv'))
df_train.head()

# %%
# transform the input data to string
df_train['tgt'] = df_train['tgt'].apply(lambda x: str(x))
df_train['src'] = df_train['src'].add(df_train['tgt'])
df_train['len'] = df_train['src'].apply(lambda x: len(x))
df_train

# %%
df_eval['tgt'] = df_eval['tgt'].apply(lambda x: str(x))
df_eval['len'] = df_eval['src'].apply(lambda x: len(x))
df_eval

# %% [markdown]
# # Build Dictionary
#  - The model cannot perform calculations directly with plain text.
#  - Convert all text (numbers/symbols) into numerical representations.
#  - Special tokens
#     - '&lt;pad&gt;'
#         - Each sentence within a batch may have different lengths.
#         - The length is padded with '&lt;pad&gt;' to match the longest sentence in the batch.
#     - '&lt;eos&gt;'
#         - Specifies the end of the generated sequence.
#         - Without '&lt;eos&gt;', the model will not know when to stop generating.

# %%
char_to_id = {}
id_to_char = {}

# write your code here
# Build a dictionary and give every token in the train dataset an id
# The dictionary should contain <eos> and <pad>
# char_to_id is to conver charactors to ids, while id_to_char is the opposite
char_to_id = {}
id_to_char = {}
chars = set()

for row in df_train.itertuples(index=True, name='Row'):
    for char in row.tgt:
        if char not in chars:
            chars.update(char)
    for char in row.src:
        if char not in chars:
            chars.update(char)

for row in df_eval.itertuples(index=True, name='Row'):
    for char in row.tgt:
        if char not in chars:
            chars.update(char)
    for char in row.src:
        if char not in chars:
            chars.update(char)

len(chars)

char_to_id = {'<pad>': 0, '<eos>': 1}
id_to_char = {0: '<pad>', 1: '<eos>'}

cur_idx = 2

for char in chars:
    char_to_id[char] = cur_idx
    id_to_char[cur_idx] = char
    cur_idx += 1

vocab_size = len(char_to_id)
print('Vocab size: {}'.format(vocab_size))

# %%
char_to_id

# %%
id_to_char

# %% [markdown]
# # Data Preprocessing
#  - The data is processed into the format required for the model's input and output. (End with \<eos\> token)
#

# %%
tqdm.pandas()
# Write your code here
df_train["char_id_list"] = None
df_train["label_id_list"] = None


def create_char_id_list(src_str, char_to_id):
    char_ids = []
    for char in src_str:
        char_ids.append(char_to_id[char])
    char_ids.append(char_to_id["<eos>"])
    return char_ids


def create_label_id_list(row, char_to_id):
    label_ids = []

    for char in row.tgt:
        label_ids.append(char_to_id[char])
    label_ids.append(char_to_id["<eos>"])

    dif_length = row.len - len(str(row.tgt))

    for _ in range(dif_length):
        label_ids.insert(0, char_to_id["<pad>"])

    return label_ids


# I used gemini due to inefficient for loop logic before, it suggested to me the use of the "apply" format for internal pandas processing
# The rest of the logic I implemented it with basic list and dictionary manipulation, so I just converted the logic to functions to work with "apply"
# In this case is "progress_apply" because I wanted to observe the progress with tqdm
df_train["char_id_list"] = df_train["src"].progress_apply(
    lambda x: create_char_id_list(x, char_to_id))

df_train["label_id_list"] = df_train.progress_apply(
    lambda row: create_label_id_list(row, char_to_id), axis=1)

df_train.head()

# %%
tqdm.pandas()

df_eval["char_id_list"] = None
df_eval["label_id_list"] = None


def create_char_id_list(src_str, char_to_id):
    char_ids = []
    for char in src_str:
        char_ids.append(char_to_id[char])
    char_ids.append(char_to_id["<eos>"])
    return char_ids


def create_label_id_list(row, char_to_id):
    label_ids = []

    for char in row.tgt:
        label_ids.append(char_to_id[char])
    label_ids.append(char_to_id["<eos>"])

    dif_length = row.len - len(str(row.tgt))

    for _ in range(dif_length):
        label_ids.insert(0, char_to_id["<pad>"])

    return label_ids


# I used gemini due to inefficient for loop logic before, it suggested to me the use of the "apply" format for internal pandas processing
# The rest of the logic I implemented it with basic list and dictionary manipulation, so I just converted the logic to functions to work with "apply"
# In this case is "progress_apply" because I wanted to observe the progress with tqdm
df_eval["char_id_list"] = df_eval["src"].progress_apply(
    lambda x: create_char_id_list(x, char_to_id))

df_eval["label_id_list"] = df_eval.progress_apply(
    lambda row: create_label_id_list(row, char_to_id), axis=1)

df_eval.head()

# %% [markdown]
# # Hyper Parameters
#
# |Hyperparameter|Meaning|Value|
# |-|-|-|
# |`batch_size`|Number of data samples in a single batch|64|
# |`epochs`|Total number of epochs to train|10|
# |`embed_dim`|Dimension of the word embeddings|256|
# |`hidden_dim`|Dimension of the hidden state in each timestep of the LSTM|256|
# |`lr`|Learning Rate|0.001|
# |`grad_clip`|To prevent gradient explosion in RNNs, restrict the gradient range|1|

# %%
batch_size = 128
epochs = 10
embed_dim = 512
hidden_dim = 512
lr = 0.0005
grad_clip = 1

# %% [markdown]
# # Data Batching
# - Use `torch.utils.data.Dataset` to create a data generation tool called  `dataset`.
# - The, use `torch.utils.data.DataLoader` to randomly sample from the `dataset` and group the samples into batches.
#
# - Example: 1+2-3=0
#     - Model input: 1 + 2 - 3 = 0
#     - Model output: / / / / / 0 &lt;eos&gt;  (the '/' can be replaced with &lt;pad&gt;)
#     - The key for the model's output is that the model does not need to predict the next character of the previous part. What matters is that once the model sees '=', it should start generating the answer, which is '0'. After generating the answer, it should also generate&lt;eos&gt;

# %%


class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        # return the amount of data
        return len(self.sequences)  # Write your code here

    def __getitem__(self, index):
        # Extract the input data x and the ground truth y from the data
        x = self.sequences.iloc[index]["char_id_list"]  # Write your code here
        y = self.sequences.iloc[index]["label_id_list"]  # Write your code here
        return x, y

# collate function, used to build dataloader


def collate_fn(batch):
    batch_x = [torch.tensor(data[0]) for data in batch]
    batch_y = [torch.tensor(data[1]) for data in batch]
    batch_x_lens = torch.LongTensor([len(x) for x in batch_x])
    batch_y_lens = torch.LongTensor([len(y) for y in batch_y])

    # Pad the input sequence
    pad_batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,
                                                  batch_first=True,
                                                  padding_value=char_to_id['<pad>'])

    pad_batch_y = torch.nn.utils.rnn.pad_sequence(batch_y,
                                                  batch_first=True,
                                                  padding_value=char_to_id['<pad>'])

    return pad_batch_x, pad_batch_y, batch_x_lens, batch_y_lens


# %%
ds_train = Dataset(df_train[['char_id_list', 'label_id_list']])

# %%
# Build dataloader of train set and eval set, collate_fn is the collate function
dl_train = torch.utils.data.DataLoader(dataset=ds_train, collate_fn=collate_fn,
                                       batch_size=batch_size, shuffle=True, num_workers=32)  # Write your code here

# %%
ds_eval = Dataset(df_eval[['char_id_list', 'label_id_list']])
dl_eval = torch.utils.data.DataLoader(dataset=ds_eval, collate_fn=collate_fn,
                                      batch_size=batch_size, shuffle=False, num_workers=32)  # Write your code here

# %% [markdown]
# # Model Design
#
# ## Execution Flow
# 1. Convert all characters in the sentence into embeddings.
# 2. Pass the embeddings through an LSTM sequentially.
# 3. The output of the LSTM is passed into another LSTM, and additional layers can be added.
# 4. The output from all time steps of the final LSTM is passed through a Fully Connected layer.
# 5. The character corresponding to the maximum value across all output dimensions is selected as the next character.
#
# ## Loss Function
# Since this is a classification task, Cross Entropy is used as the loss function.
#
# ## Gradient Update
# Adam algorithm is used for gradient updates.

# %%


class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharRNN, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embed_dim,
                                            padding_idx=char_to_id['<pad>'])

        self.rnn_layer1 = torch.nn.LSTM(input_size=embed_dim,
                                        hidden_size=hidden_dim,
                                        batch_first=True)

        self.rnn_layer2 = torch.nn.LSTM(input_size=hidden_dim,
                                        hidden_size=hidden_dim,
                                        batch_first=True)

        self.linear = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_dim,
                                                          out_features=hidden_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(in_features=hidden_dim,
                                                          out_features=vocab_size))

    def forward(self, batch_x, batch_x_lens):
        return self.encoder(batch_x, batch_x_lens)

    # The forward pass of the model
    def encoder(self, batch_x, batch_x_lens):
        batch_x = self.embedding(batch_x)

        batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch_x,
                                                          batch_x_lens,
                                                          batch_first=True,
                                                          enforce_sorted=False)

        batch_x, _ = self.rnn_layer1(batch_x)
        batch_x, _ = self.rnn_layer2(batch_x)

        batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x,
                                                            batch_first=True)

        batch_x = self.linear(batch_x)

        return batch_x

    def generator(self, start_char, max_len=200):

        char_list = [char_to_id[c] for c in start_char]

        next_char = None

        while len(char_list) < max_len:
            # Write your code here
            # Pack the char_list to tensor
            device = self.embedding.weight.device
            input_tensor = torch.tensor(char_list).unsqueeze(0).to(device)

            # Input the tensor to the embedding layer, LSTM layers, linear respectively
            embedding_input = self.embedding(input_tensor)
            embedding_input, _ = self.rnn_layer1(embedding_input)
            embedding_input, _ = self.rnn_layer2(embedding_input)
            # Obtain the next token prediction y
            y = self.linear(embedding_input)
            last_char_logits = y[:, -1, :]
            # Use argmax function to get the next token prediction
            next_char = torch.argmax(last_char_logits, dim=1).item()

            if next_char == char_to_id['<eos>']:
                break

            char_list.append(next_char)

        return [id_to_char[ch_id] for ch_id in char_list]


# %%
torch.manual_seed(2)


# Write your code here. Specify a device (cuda or cpu)
device = torch.device("cuda")

model = CharRNN(vocab_size,
                embed_dim,
                hidden_dim)

# %%
# Write your code here. Cross-entropy loss function. The loss function should ignore <pad>
criterion = torch.nn.CrossEntropyLoss(
    reduction='none', ignore_index=char_to_id['<pad>'])
# Write your code here. Use Adam or AdamW for Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# %% [markdown]
# # Training
# 1. The outer `for` loop controls the `epoch`
#     1. The inner `for` loop uses `data_loader` to retrieve batches.
#         1. Pass the batch to the `model` for training.
#         2. Compare the predicted results `batch_pred_y` with the true labels `batch_y` using Cross Entropy to calculate the loss `loss`
#         3. Use `loss.backward` to automatically compute the gradients.
#         4. Use `torch.nn.utils.clip_grad_value_` to limit the gradient values between `-grad_clip` &lt; and &lt; `grad_clip`.
#         5. Use `optimizer.step()` to update the model (backpropagation).
# 2.  After every `1000` batches, output the current loss to monitor whether it is converging.

# %%
# In here I needed some help from Gemini to see how to make the mask to not pass over the characters before '=' to the loss function
# Also to make my data be in the correct tensor dimension
model = model.to(device)

i = 0
batches_loss = pd.DataFrame(columns=["epoch", "batch_no", "loss"])
epoch_val_acc = pd.DataFrame(columns=["epoch", "accuracy"])

for epoch in range(1, epochs+1):
    # The process bar
    bar = tqdm(dl_train, desc=f"Train epoch {epoch}")
    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        # Write your code here
        model.train()
        # Clear the gradient
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        batch_pred_y = model(batch_x, batch_x_lens)

        # Write your code here
        # Input the prediction and ground truths to loss function

        labels_reshaped = batch_y[:, 1:].contiguous().view(-1)
        predictions_reshaped = batch_pred_y[:, :-1,
                                            :].contiguous().view(-1, vocab_size).to(device)

        # Create mask
        equal_token_id = char_to_id['=']
        mask = torch.zeros_like(batch_y, dtype=torch.bool, device=device)
        for b in range(batch_y.size(0)):
            eq_positions = (batch_x[b] == equal_token_id).nonzero(
                as_tuple=True)[0]
            if len(eq_positions) > 0:
                eq_idx = eq_positions[0].item()
                # the model predicts from eq_idx+1 onward
                mask[b, eq_idx + 1: batch_y_lens[b]] = True

        # compute loss on mask
        mask = mask[:, 1:]

        # criterion = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=char_to_id['<pad>'])
        loss_all = criterion(predictions_reshaped, labels_reshaped)
        loss_all = loss_all.view(mask.size())

        # Apply mask
        masked_loss = loss_all * mask.float()
        loss = masked_loss.sum() / mask.sum()

        # Back propagation
        loss.backward()

        torch.nn.utils.clip_grad_value_(
            model.parameters(), grad_clip)  # gradient clipping

        # Write your code here
        # Optimize parameters in the model
        optimizer.step()

        i += 1
        if i % 50 == 0:
            bar.set_postfix(loss=loss.item())

        if i == 1:
            batches_loss = pd.concat([batches_loss, pd.DataFrame(
                [[epoch, i, loss.item()]], columns=["epoch", "batch_no", "loss"])], ignore_index=True)

        # Prints loss every 1000 batches
        if i % 1000 == 0:
            batches_loss = pd.concat([batches_loss, pd.DataFrame(
                [[epoch, i, loss.item()]], columns=["epoch", "batch_no", "loss"])], ignore_index=True)
            # print(f"loss: {loss.item()}")

    # Evaluate your model
    matched = 0
    total = 0
    count = 0
    # model.eval()
    with torch.no_grad():
        bar_eval = tqdm(dl_eval, desc=f"Validation epoch {epoch}")
        for batch_x, batch_y, batch_x_lens, batch_y_lens in bar_eval:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            for b in range(batch_x.size(0)):
                total += 1

                # An example of using generator: model.generator('1+1=')

                # Write your code here. Input the batch_x to the model and generate the predictions
                input_chars = [id_to_char[idx.item()]
                               for idx in batch_x[b] if idx.item() != char_to_id['<pad>']]
                input_str = "".join(input_chars).replace("<eos>", "")
                # print(input_str)
                prediction = model.generator(input_str)
                prediction_str = "".join(prediction)

                if '=' in prediction_str:
                    predicted_answer = prediction_str.split('=')[1].strip()
                else:
                    predicted_answer = prediction_str.strip()

                # prediction_reshaped = prediction.view(-1, vocab_size)
                # label_reshaped = batch_y.to(device).view(-1)

                # Write your code here.
                # Check whether the prediction match the ground truths
                # Compute exact match (EM) on the eval dataset
                # EM = correct/total

                target_chars = [id_to_char[idx.item()] for idx in batch_y[b] if idx.item() not in [
                    char_to_id['<pad>'], char_to_id['<eos>']]]
                target_str = "".join(target_chars).strip()

                if predicted_answer == target_str:
                    matched += 1
                # Printing some prediction vs groundtruth evaluation for debugging and visualization
                if count < 5:
                    print(f"prediction: {predicted_answer}")
                    print(f"batch_y: {target_str}")
                    count += 1

        print(f"EM Accuracy: {matched/total}")
        epoch_val_acc = pd.concat([epoch_val_acc, pd.DataFrame(
            [[epoch, matched/total]], columns=["epoch", "accuracy"])], ignore_index=True)

# %%
# Here after structuring the data, gemini helped me create a good plot when showing some examples online on how to use double axis in plots and overlaying different dataframes in the figure
output_dir = "./results/results_comb_hyp_2"
os.makedirs(output_dir, exist_ok=True)

batches_loss['global_batch'] = range(len(batches_loss))

# Calculate the position for epoch markers
epoch_boundaries = batches_loss.groupby('epoch')['global_batch'].min().tolist()
epoch_boundaries.append(len(batches_loss))

# Get the center of each epoch
epoch_centers = [(epoch_boundaries[i] + epoch_boundaries[i+1]
                  ) / 2 for i in range(len(epoch_boundaries)-1)]


# Create the plot
fig, ax1 = plt.subplots(figsize=(15, 7))

# Plot batch loss on the primary y-axis
color = 'tab:red'
ax1.set_xlabel('Global Batch Number')
ax1.set_ylabel('Batch Loss', color=color)
sns.lineplot(data=batches_loss, x='global_batch', y='loss',
             ax=ax1, color=color, alpha=0.7, label='Batch Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(bottom=min(batches_loss["loss"]))

# Create a secondary y-axis for accuracy
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Validation Accuracy', color=color)
sns.lineplot(data=epoch_val_acc, x=epoch_centers, y='accuracy', ax=ax2,
             color=color, marker='o', linestyle='--', label='Validation Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(min(epoch_val_acc["accuracy"])-0.05,
             max(epoch_val_acc["accuracy"])+0.05)

# Add vertical lines and text for epoch boundaries
for i, boundary in enumerate(epoch_boundaries[:-1]):
    ax1.axvline(x=boundary, color='gray', linestyle='--', linewidth=1)
    # Add text label for the epoch
    ax1.text(epoch_centers[i], ax1.get_ylim()[
             1] * 0.90, f'Epoch {i+1}\nAcc={epoch_val_acc.iloc[i]["accuracy"]:.3f}', horizontalalignment='center', color='black')

plt.title('Results With LSTM - 10 Epochs - 512 Emb & Hidden Dim - 128 Batch Size - 0.0005 LR - Batch Loss and Validation Accuracy Over Training')
fig.tight_layout()
plt.savefig(f"{output_dir}/loss_and_acc_model.png")
plt.show()

# %%
batches_loss.to_csv(f"{output_dir}/batches_loss.csv")
epoch_val_acc.to_csv(f"{output_dir}/epoch_val_acc.csv")
