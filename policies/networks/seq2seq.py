# eager version of sequence to sequence network
import tensorflow as tf
import numpy as np
import random

import tensorflow_probability as tfp

def gru(units):
  # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
  # the code automatically does that.
  if tf.test.is_gpu_available():
    return tf.keras.layers.CuDNNGRU(units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
  else:
    return tf.keras.layers.GRU(units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_activation='sigmoid',
                               recurrent_initializer='glorot_uniform')

class Encoder(tf.keras.Model):
    def __init__(self, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        # bidirected gru
        self.gru = tf.keras.layers.Bidirectional(gru(self.enc_units))
        self.batch_norm = tf.keras.layers.BatchNormalization()


    def call(self, x):
        x = self.batch_norm(x)
        output, fw_state, bw_state = self.gru(x)

        state = tf.compat.v1.concat([fw_state, bw_state], axis=-1)

        return output, state

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    hidden_with_time_axis = tf.expand_dims(query, 1)

    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units):
    super(Decoder, self).__init__()
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = gru(self.dec_units)
    self.fc = tf.keras.layers.Dense(vocab_size)
    self.batch_norm = tf.keras.layers.BatchNormalization()

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # add layer normalization to the task
    x = self.batch_norm(x)
    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

class Seq2SeqNetwork(tf.keras.Model):
    def __init__(self, encoder_units, decoder_units,
                 vocab_size):
        super(Seq2SeqNetwork, self).__init__()
        self.encoder = Encoder(enc_units=encoder_units)

        self.decoder = Decoder(dec_units=decoder_units, vocab_size=vocab_size,
                               embedding_dim=decoder_units)

    def call(self, enc_input, targ):
        enc_output, enc_hidden = self.encoder(enc_input)
        dec_hidden = enc_hidden

        decoder_output_logits = []
        for t in range(0, targ.shape[1]):
            dec_input = tf.expand_dims(targ[:, t], 1)
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                      dec_hidden,
                                                                      enc_output)

            predicted_logits = predictions
            decoder_output_logits.append(predicted_logits)

        decoder_output_logits = tf.stack(decoder_output_logits, axis=1)

        return decoder_output_logits


    def sample(self, x):
        batch_size = x.shape[0]
        seqence_length = x.shape[1]

        enc_output, enc_hidden = self.encoder(x)

        dec_hidden = enc_hidden
        dec_input = tf.zeros((batch_size, 1))
        decoder_output_id = []
        decoder_output_logits = []

        for t in range(seqence_length):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                      dec_hidden,
                                                                      enc_output)


            # storing the attention weights to plot later on
            predicted_sampler = tfp.distributions.Categorical(logits=predictions)
            predicted_id = predicted_sampler.sample(seed=random.seed())
            #predicted_id = tf.argmax(predictions, axis=-1).numpy()
            predicted_logits = predictions

            decoder_output_id.append(predicted_id)
            decoder_output_logits.append(predicted_logits)

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims(predicted_id, axis=-1)

        decoder_output_id  = tf.stack(decoder_output_id, axis=1)
        decoder_output_logits = tf.stack(decoder_output_logits, axis=1)

        return decoder_output_id, decoder_output_logits

    def greedy_select(self, x):
        batch_size = x.shape[0]
        seqence_length = x.shape[1]

        enc_output, enc_hidden = self.encoder(x)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([0] * batch_size, 1)

        decoder_output_id = []
        decoder_output_logits = []

        for t in range(seqence_length):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                      dec_hidden,
                                                                      enc_output)

            # storing the attention weights to plot later on

            predicted_id = tf.argmax(predictions, axis=-1).numpy()
            predicted_logits = predictions

            decoder_output_id.append(predicted_id)
            decoder_output_logits.append(predicted_logits)

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims(predicted_id, 1)

        decoder_output_id = tf.stack(decoder_output_id, axis=1)
        decoder_output_logits = tf.stack(decoder_output_logits, axis=1)

        print(decoder_output_id)

        return decoder_output_id, decoder_output_logits


def copy_model(model, x):
    '''Copy model weights to a new model.

    Args:
        model: model to be copied.
        x: An input example. This is used to run
            a forward pass in order to add the weights of the graph
            as variables.
    Returns:
        A copy of the model.
    '''
    copied_model = Seq2SeqNetwork(encoder_units=30, decoder_units=60,
                 vocab_size=2)

    # If we don't run this step the weights are not "initialized"
    # and the gradients will not be computed.
    copied_model(tf.convert_to_tensor(x))

    copied_model.set_weights(model.get_weights())
    return copied_model

if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()

    seq2seq_model = Seq2SeqNetwork(encoder_units=1, decoder_units=2,
                 vocab_size=2)


    input_data = np.array(np.arange(60).reshape(2,2,15),dtype=np.float32)

    output_id, logits = seq2seq_model.greedy_select(input_data)
    print(output_id)
    var_before = seq2seq_model.trainable_variables

    input_data = np.array(np.arange(160).reshape(2,4,20), dtype=np.float32)
    output_id, logits = seq2seq_model.greedy_select(input_data)
    print(output_id)
    var_end = seq2seq_model.trainable_variables


    # print(seq2seq_model.trainable_variables)
    # print(len(seq2seq_model.trainable_variables))
    #
    # print(len(seq2seq_model.layers[0].layers))
    # print(len(seq2seq_model.layers[1].layers))

    #print(seq2seq_model.layers[1].embedding)


    # copyed_model = copy_model(seq2seq_model, input_data)
    #
    # print(output_id[0])
    # print(logits[0])
