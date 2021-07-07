from rnn import RNN

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
# training_set = [sequence_to_inputs_and_targets([i, j, next(iter(set(range(3)) - set([i, j])))]) for i in range(3) for j in range(3)]
# rnn.train(training_set, 5_000)

print('Third element is the (mod) sum of the first 2:')
rnn = RNN(vocab_size=5, hidden_dim=100)
training_set = [sequence_to_inputs_and_targets([i, j, (i+j) % 5]) for i in range(5) for j in range(5)]
rnn.train(training_set, 5_000)
