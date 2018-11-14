import os
import tensorflow as tf
import bert
from data_helper import *

# TODO: support to add individual new models

class NLPEncoder(object):
    """
    1. fit all the layers of encoder, then encoder is a feature extraction;
    2. fit some certain layers and then train other layers
    """
    def __init__(self, model_name='bert', mode='test'):
        self.is_training = True if mode == 'train' else False
        if model_name == 'bert':
            self.FLAGS = get_bert_flag()
            self.create_initialize_bert(self.FLAGS.bert_config_file)

        if mode == 'train':
            # TODO: return the last layer, then you can append a task specific model on it
            return self.get_output_layer()

    def encode(self, texts_a, texts_b=None, mode='cls'):
        """
        :param text_a: a list of the first sequence
        :param text_b: (optional), a list of the second sequence
        :param mode: 'cls' or 'all', 'cls' is the first token pooled_output on the last layer of bert,
                      'all' is all layer outputs, containing embedding output(input of trasformer),
                      each layer of trasformer(12 layers in base bert and 24 layers in large bert),
                      and 'cls' output.
        :return: a list of the output of each needed layer
                 if mode is 'cls': the dimension is (sequence_id, embedding),
                 if mode is 'all': the dimension is (layer_id, sequence_id, embedding), the first dimension's shape is
                                   15 in base bert and 27 in large bert.
        """
        # TODO: do some text preprocessing here

        input_ids, input_mask, segment_ids, _ = TextProcessor.get_processed_ids(self.FLAGS, texts_a, texts_b)
        output_layers = self.get_layers()
        if mode == 'cls':
            output_layers = output_layers[-1]

        return self.sess.run(output_layers, feed_dict={'input_ids:0': input_ids})


    def train(self):
        model = self.model


    def test_bert(self, model_config):
        import bert
        import tensorflow as tf
        model_config = './model/cased_L-12_H-768_A-12/bert_config.json'
        init_checkpoint = './model/cased_L-12_H-768_A-12/bert_model.ckpt'

        # with tf.Session() as sess:
        sess = tf.Session()
        init_vars = tf.train.list_variables(init_checkpoint)
        kernel = tf.train.load_variable(init_checkpoint, 'bert/encoder/layer_0/output/dense/kernel')
        bert_config = bert.BertConfig.from_json_file(model_config)
        model = bert.BertModel(config=bert_config, is_training=False)

        tvars = tf.trainable_variables()
        assignment_map, initialized_variable_names = bert.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        sess.run(tf.global_variables_initializer())

        # ids = sess.run(['input_ids:0'])
        # output = sess.run([model.get_pooled_output()])
        # kernel0 = sess.run(['bert/encoder/layer_0/output/dense/kernel:0'])

        # sess.run([model.get_pooled_output()], feed_dict={'input_ids:0':[[1,2,3]]})

    def create_initialize_bert(self, model_config):
        tf.logging.set_verbosity(tf.logging.WARN)
        self.sess = tf.Session()
        bert_config = bert.BertConfig.from_json_file(model_config)
        self.model = bert.BertModel(config=bert_config, is_training=self.is_training)

        tvars = tf.trainable_variables()
        assignment_map, initialized_variable_names = bert.get_assignment_map_from_checkpoint(tvars,
                                                                                             self.FLAGS.init_checkpoint)
        tf.train.init_from_checkpoint(self.FLAGS.init_checkpoint, assignment_map)
        self.sess.run(tf.global_variables_initializer())
        # tf.logging.info("**** Trainable Variables ****")
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    def get_layers(self, layers=None):
        """
        1. fit some layers, and train the other layers
        2. append classifier on one middle certain layer
        :return: 14 layers in base and 26 layers in large
        """
        output_layers = []

        # the layer your task-specific model build on
        embedding_layer_output = self.model.get_embedding_output()  # 1 layer
        trasformer_layers_output = self.model.get_all_encoder_layers()  # 12 layers in base or 24 layers in large
        cls_output = self.model.get_pooled_output()  # 1 layer

        output_layers.append(embedding_layer_output)
        output_layers.extend(trasformer_layers_output)
        output_layers.append(cls_output)

        return output_layers