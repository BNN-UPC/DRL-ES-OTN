# -*- coding: utf-8 -*-
# Copyright (c) 2022, Carlos Güemes [^1], Paul Almasan [^2]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: carlos.guemes@estudiantat.upc.edu
# [^2]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: felician.paul.almasan@upc.edu

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import regularizers

class myModel(tf.keras.Model):
    def __init__(self, hparams, hidden_init_actor, kernel_init_actor):
        super(myModel, self).__init__()
        self.hparams = hparams

        # Define layers here
        # Message functions
        self.Message = tf.keras.models.Sequential()
        self.Message.add(keras.layers.Dense(self.hparams['link_state_dim'],
                                            kernel_initializer=hidden_init_actor,
                                            activation=tf.nn.selu, name="FirstLayer",
                                            dtype=tf.float32))
        # Update function (RNN)
        self.Update = tf.keras.layers.GRUCell(self.hparams['link_state_dim'], dtype=tf.float32)

        # Readout function (global summary)
        self.Readout = tf.keras.models.Sequential()
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_initializer=hidden_init_actor,
                                            name="Readout1",
                                            dtype=tf.float32))
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_initializer=hidden_init_actor,
                                            name="Readout2",
                                            dtype=tf.float32))
        self.Readout.add(keras.layers.Dense(1, kernel_initializer=kernel_init_actor, name="Readout3", dtype=tf.float32))

    def build(self, input_shape=None):
        # Create the weights of the layer
        self.Message.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim']*2]))
        self.Update.build(input_shape=tf.TensorShape([None,self.hparams['link_state_dim']]))
        self.Readout.build(input_shape=[None, self.hparams['link_state_dim']])
        self.built = True

    #@tf.function
    def call(self, link_state, states_graph_ids, states_first, states_second, states_num_edges):

        # Execute T times
        for _ in range(self.hparams['T']):
            # We have the combination of the hidden states of the main edges with the neighbours
            # Features of the links
            mainEdges = tf.gather(link_state, states_first)
            #Features of the neighboring links
            neighEdges = tf.gather(link_state, states_second)
            # This values represent a double loop, where for each row we will represent a link and its neighbors
            # This set up allows us to quickly consider all links and its neihbors.
            # Therefore, the sum in line 5 of Algorithm 1 would be represented as sef.Message(mainEdges, neighEdges)
            # Sum inlcuded

            edgesConcat = tf.concat([mainEdges, neighEdges], axis=1)

            ### 1.a Message passing for link with all it's neighbours
            outputs = self.Message(edgesConcat)

            ### 1.b Sum of output values according to link id index
            # states_num_edges allows to segment the sum for each individual graph
            edges_inputs = tf.math.unsorted_segment_sum(data=outputs, segment_ids=states_second,
                                                        num_segments=states_num_edges)

            ### 2. Update for each link
            # GRUcell needs a 3D tensor as state because there is a matmul: Wrap the link state
            outputs, links_state_list = self.Update(edges_inputs, [link_state])

            link_state = links_state_list[0]

        # Perform sum of all hidden states
        # Done separetely for each graph
        edges_combi_outputs = tf.math.segment_sum(link_state, states_graph_ids, name=None)

        r = self.Readout(edges_combi_outputs)
        return r

    @staticmethod
    def _get_specific_number_weights(model):
        weights = model.get_weights()
        layer_dimensions = [(w.shape, w.size) for w in weights]
        return layer_dimensions, sum(w[1] for w in layer_dimensions)

    def get_message_number_weights(self):
        return self._get_specific_number_weights(self.Message)

    def get_update_number_weights(self):
        return self._get_specific_number_weights(self.Update)

    def get_message_update_number_weights(self):
        message_layer_dimensions, message_number_params = self._get_specific_number_weights(self.Message)
        update_layer_dimensions, update_number_params = self._get_specific_number_weights(self.Update)
        return message_layer_dimensions + update_layer_dimensions, message_number_params + update_number_params

    def get_readout_number_weights(self):
        return self._get_specific_number_weights(self.Readout)

    def get_number_weights(self):
        return self._get_specific_number_weights(super(myModel, self))

    @staticmethod
    def _get_specific_weights(model):
        weights = model.get_weights()
        for w in range(len(weights)):
            weights[w] = np.reshape(weights[w], (weights[w].size,))
        return np.concatenate(weights)

    def get_message_weights(self):
        return self._get_specific_weights(self.Message)

    def get_update_weights(self):
        return self._get_specific_weights(self.Update)

    def get_message_update_weights(self):
        return np.concatenate((self._get_specific_weights(self.Message), self._get_specific_weights(self.Update)))

    def get_readout_weights(self):
        return self._get_specific_weights(self.Readout)

    def get_weights(self):
        return self._get_specific_weights(super(myModel, self))

    @staticmethod
    def _set_weights(model, new_weights):
        weights = model.get_weights()
        layer_dimensions = [(w.shape, w.size) for w in weights]

        transformed_weights = []
        current_idx = 0
        for layer_shape, layer_size in layer_dimensions:
            layer_weights = np.reshape(new_weights[current_idx:current_idx + layer_size], layer_shape)
            transformed_weights.append(layer_weights)
            current_idx += layer_size

        model.set_weights(transformed_weights)

    def set_message_weights(self, new_weights):
        self._set_weights(self.Message, new_weights)

    def set_update_weights(self, new_weights):
        self._set_weights(self.Update, new_weights)

    def set_message_update_weights(self, new_weights):
        _, message_number_params = self.get_message_number_weights()
        self._set_weights(self.Message, new_weights[:message_number_params])
        self._set_weights(self.Update, new_weights[message_number_params:])

    def set_readout_weights(self, new_weights):
        self._set_weights(self.Readout, new_weights)

    def set_weights(self, new_weights):
        self._set_weights(super(myModel, self), new_weights)




