from numpy import random
from rnn import RNN

from itertools import permutations

def sequence_to_inputs_and_targets(sequence):
    return sequence[:-1], sequence[1:]

# print('Constant 0:')
# rnn = RNN(vocab_size=3, hidden_dim=5)
# rnn.train([sequence_to_inputs_and_targets([0] * 100)])

# print('Constant 1:')
# rnn = RNN(vocab_size=3, hidden_dim=5)
# rnn.train([sequence_to_inputs_and_targets([1] * 100)])

# print('Constant 2:')
# rnn = RNN(vocab_size=3, hidden_dim=5)
# rnn.train([sequence_to_inputs_and_targets([2] * 100)])

# print('0, 1, ...:')
# rnn = RNN(vocab_size=2, hidden_dim=10)
# rnn.train([sequence_to_inputs_and_targets([0, 1] * 100)])

# print('0, 1, ... (larger alphabet):')
# rnn = RNN(vocab_size=10, hidden_dim=10)
# rnn.train([sequence_to_inputs_and_targets([0, 1] * 100)])

# print('0, 1, 2, ... (larger alphabet):')
# rnn = RNN(vocab_size=10, hidden_dim=10)
# rnn.train([sequence_to_inputs_and_targets([0, 1, 2] * 100)])

# print('0, 1, 2, 1, 0, ...:')
# rnn = RNN(vocab_size=10, hidden_dim=20)
# rnn.train([sequence_to_inputs_and_targets([0, 1, 2, 1, 0] * 100)])

# print('First element determines the rest:')
# rnn = RNN(vocab_size=10, hidden_dim=50)
# rnn.train([sequence_to_inputs_and_targets([i] * (30+i*5)) for i in range(10)], 2_000, optimizer_name='gd', learning_rate=1e-3, plot_color='b')

# print('First element determines the rest:')
# rnn = RNN(vocab_size=10, hidden_dim=50)
# rnn.train([sequence_to_inputs_and_targets([i] * (30+i*5)) for i in range(10)], 2_000, optimizer_name='adagrad', plot_color='r')

# print('First element determines the rest:')
# rnn = RNN(vocab_size=10, hidden_dim=50)
# rnn.train([sequence_to_inputs_and_targets([i] * (30+i*5)) for i in range(10)], 2_000, optimizer_name='rprop', learning_rate=1e-2, plot_color='g')

# print('Third element is the one not appearing in the first 2:')
# rnn = RNN(vocab_size=3, hidden_dim=50)
# training_set = [sequence_to_inputs_and_targets(perm) for perm in permutations(range(3))]
# rnn.train(training_set, 5_000, sample_prefix_length=2)

# print('Third element is the one not appearing in the first 3:')
# rnn = RNN(vocab_size=4, hidden_dim=5)
# training_set = [sequence_to_inputs_and_targets(perm) for perm in permutations(range(4))]
# rnn.train(training_set, 5_000, sample_prefix_length=3)

# print('Third element is the (mod) sum of the first 2:')
# rnn = RNN(vocab_size=5, hidden_dim=100)
# training_set = [sequence_to_inputs_and_targets([i, j, (i+j) % 5]) for i in range(5) for j in range(5)]
# rnn.train(training_set, 5_000, sample_prefix_length=2)

# print('Zigzaging starting with first element')
# rnn = RNN(vocab_size=2, hidden_dim=20)
# training_set = list(map(sequence_to_inputs_and_targets, ([0, 1] * 10, [1, 0] * 10)))
# rnn.train(training_set, 5_000)

# print('First element selects complex sequence')
# rnn = RNN(vocab_size=2, hidden_dim=512)
# SEQ_LEN = 20
# NUM_SEQS = 1
# sequences = [[random.choice([0, 1]) for _ in range(SEQ_LEN)] for _ in range(NUM_SEQS)]
# training_set = list(map(sequence_to_inputs_and_targets, sequences))
# # rnn.train(training_set, 10_000, sample_prefix_length=SEQ_LEN // 2)
# rnn.train(training_set, 5_000)

def load_text_from_file(filename, max_length:int = -1):
    text = open(filename).read()[:max_length]
    chars = list(set(text))
    print(f'Total {len(text)} characters ({len(chars)} vocab size)')
    char_to_index = {c:i for i, c in enumerate(chars)}
    index_to_char = {v:k for k, v in char_to_index.items()}
    return [char_to_index[c] for c in text], char_to_index, index_to_char

print('Generating Tarantino...')
data, char_to_index, index_to_char = load_text_from_file('./pulp_fiction.txt', 10_000)
rnn = RNN(vocab_size=len(char_to_index), hidden_dim=100)
training_set = [sequence_to_inputs_and_targets(data)]
rnn.train(training_set, 5_000, sample_tokenizer=lambda i: index_to_char[i])
