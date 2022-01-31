# -*- coding: utf-8 -*-
# Copyright (c) 2021, Paul Almasan [^1]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: felician.paul.almasan@upc.edu

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
                                            activation=tf.nn.selu, name="FirstLayer"))
        # Update function (RNN)
        self.Update = tf.keras.layers.GRUCell(self.hparams['link_state_dim'], dtype=tf.float32)

        # Readout function (global summary)
        self.Readout = tf.keras.models.Sequential()
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_initializer=hidden_init_actor,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout1"))
        self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_initializer=hidden_init_actor,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout2"))
        self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(keras.layers.Dense(1, kernel_initializer=kernel_init_actor, name="Readout3"))

    def build(self, input_shape=None):
        # Create the weights of the layer
        self.Message.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim']*2]))
        self.Update.build(input_shape=tf.TensorShape([None,self.hparams['link_state_dim']]))
        self.Readout.build(input_shape=[None, self.hparams['link_state_dim']])
        self.built = True

    #@tf.function
    def call(self, link_state, states_graph_ids, states_first, states_second, states_num_edges, training=False):

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

        r = self.Readout(edges_combi_outputs,training=training)
        return r