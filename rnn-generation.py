import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time

class RNN_Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):

        rnn_var = self.embedding(inputs, training=training)
        if states is None:
          states = self.gru.get_initial_state(rnn_var)
        rnn_var, states = self.gru(rnn_var, initial_state=states, training=training)
        rnn_var = self.dense(rnn_var, training=training)

        if return_state:
          return rnn_var, states
        else:
          return rnn_var




class RNN_Text_Generation(object):

    def __init__(self):

        self.config_args = {
            "file" : "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
            "epochs" : 30,
            "sequence_length" : 100,
            "batch_size" : 64,
            "buffer_size" : 10000,
            "rnn_units" : 1024,
            "embedding_dimensions" : 256,
            "encoding" : "utf-8",
            "optimizer" : "adam",
        }

    def get_text(self):

        if "http" in self.config_args["file"]: # If the file seems to be a link
            name = self.config_args["file"].split('/')[-1] # get the last instance after a "/"

            file_path = tf.keras.utils.get_file(name, self.config_args["file"])

        else:
            file_path = self.config_args["file"]

        with open(file_path, 'rb') as file:
            text = file.read().decode(encoding=self.config_args["encoding"]) # The text to parse
            file.close()

        return text

    def rnn_preprocessing(self, text):

       self.vocab = sorted(set(text))

        self.ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))

        chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=self.ids_from_chars.get_vocabulary(), invert=True)

        def text_from_ids(ids):
          return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

        # training
        all_ids = ids_from_chars(tf.strings.unicode_split(text, self.config_args["encoding"]'))
        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

        seq_length = self.config_args["sequence_length"]
        examples_per_epoch = len(text)//(seq_length+1)

        sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

        def split_input_target(sequence):
            input_text = sequence[:-1]
            target_text = sequence[1:]
            return input_text, target_text

        self.dataset = sequences.map(split_input_target)

    def batching(self, dataset):
        self.rnn_preprocessing(self.get_text())

        self.dataset = (
            self.dataset
            .shuffle(self.config_args["buffer_size"])
            .batch(self.config_args["batch_size"], drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )


    def gen_model(self, ids_from_chars):

        self.model = RNN_Model(
            # Be sure theself.vocabulary size matches the `StringLookup` layers.
            vocab_size=len(self.ids_from_chars.get_vocabulary()),
            embedding_dim=self.config_args["embedding_dimensions"],
            rnn_units=self.config_args["rnn_units"]

            )

        for input_example_batch, target_example_batch in self.dataset.take(1):
            example_batch_predictions = self.model(input_example_batch)

        self.model.summary()

        return self.model

    def training(self):
        # training process

        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

        example_batch_loss = loss(target_example_batch, example_batch_predictions)
        mean_loss = example_batch_loss.numpy().mean()
        print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
        print("Mean loss:        ", mean_loss)

        tf.exp(mean_loss).numpy()
        model.compile(optimizer=self.config_args["optimizer"], loss=loss)

        # Directory where the checkpoints will be saved
        checkpoint_dir = './training_checkpoints'
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)


        EPOCHS = 20

        history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
