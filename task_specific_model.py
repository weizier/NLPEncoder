import tensorflow as tf
from data_helper import *
import time
import random
from encoder import NLPEncoder

# TODO: Squad, seq2seq, crf, qa,

class TaskSpecificModel:
    def __init__(self, encoder):
        pass

    def get_encoder_output(self):
        raise NotImplementedError("You must implement encoder function.")

    def get_output_layer(self):
        raise NotImplementedError("You must implement output function.")


class Classifier(TaskSpecificModel):

    def __init__(self, path_or_data, encoder='bert', language='en', col_num=2, encoder_layer='last'):
        self.FLAGS = get_classifier_flag()
        self.encoder = self.get_encoder(encoder, language)
        self.dropout_keep_prob = self.encoder.model.dropout_keep_prob
        self.sess = self.encoder.sess
        # data_processor = MrpcProcessor(self.encoder.FLAGS)
        # path_or_data = data_processor
        self.build_data_manager(path_or_data, col_num)  # you can change the data manager when training
        self.create_placeholders()

        with tf.variable_scope("classifier"):
            self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
            self.encoder_output = self.get_encoder_output(mode=encoder_layer)
            self.features = self.get_features(mode='mlp')
            self.logits = self.build_output_layer()

            self.compute_loss_acc()
            self.get_train_op()

    def get_encoder(self, encoder='bert', language='en'):
        return NLPEncoder(FLAGS=self.FLAGS, model_name=encoder, language=language)

    def create_placeholders(self):
        self.y = tf.placeholder(dtype=tf.int64, shape=None, name="y")

    def build_data_manager(self, path_or_data, col_num):
        assert isinstance(path_or_data, (str, list, MnliProcessor, ColaProcessor, MrpcProcessor, ClassifierProcessor,
                                         XnliProcessor)), "Make sure path_or_data is a filepath or list of data!"

        if isinstance(path_or_data, (MnliProcessor, ColaProcessor, MrpcProcessor, ClassifierProcessor, XnliProcessor)):
            self.dm = path_or_data
        if isinstance(path_or_data, str):
            self.dm = ClassifierProcessor(self.encoder.FLAGS, path_or_data, col_num)
        if isinstance(path_or_data, list):
            if col_num == 2:
                texts_a, labels = zip(*path_or_data)
                texts_b = None
            elif col_num == 3:
                texts_a, texts_b, labels = zip(*path_or_data)
            else:
                raise ValueError
            self.dm = TextProcessor(self.encoder.FLAGS, texts_a, texts_b, labels)


    def get_encoder_output(self, mode='attention'):
        """
        :param mode: 'attention' is a softmax weight on all of the layers;
                      'last' just use the last layer of encoder(but 'cls' token in bert)
                      'let_me_do': a list of all layers in encoder will return, combination will let you make it.
        :return: a combination of all encoders or a list of all layers in encoder
        """
        self.encoder_layers = self.encoder.get_layers()

        assert mode in ['attention', 'last', 'let_me_do'], "Specify a mode in ['attention', 'last', 'let_me_do']"
        if mode == 'let_me_do':
            return self.encoder_layers   # [13 * (batch_size, seq_length, hidden_size) + (batch_size, hidden_size)]
        elif mode == 'last':
            return tf.expand_dims(self.encoder_layers[-1], 1)  # (batch_size, 1, hidden_size)

        with tf.variable_scope("attention_combine"):
            hidden_size = self.encoder_layers[0].shape[-1].value

            combine_weights = tf.get_variable(
                "attention_combine_weights", [hidden_size, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            combine_bias = tf.get_variable(
                "attention_combine_bias", [hidden_size], initializer=tf.zeros_initializer())
            # [13 * (batch_size, seq_length, hidden_size)] --> (batch_size, seq_length, 13, hidden_size)
            self.combine_in = tf.concat([tf.expand_dims(layer, axis=2) for layer in self.encoder_layers[:-1]], axis=2)
            combine_in_shape = tf.shape(self.combine_in)# self.combine_in.shape
            self.combine_e = tf.nn.relu(tf.nn.xw_plus_b(tf.reshape(self.combine_in, [-1, hidden_size]),
                                                        combine_weights, combine_bias))
            self.combine_e = tf.reshape(self.combine_e, combine_in_shape)

            self.combine_e = tf.nn.dropout(self.combine_e, keep_prob=self.dropout_keep_prob)

            # (batch_size, seq_length, 13, hidden_size) --> (batch_size, seq_length, hidden_size)
            encoder_output = tf.reduce_sum(tf.nn.softmax(self.combine_e, 2) * self.combine_in, axis=2)
            return encoder_output  # (batch_size, seq_length, hidden_size)

    def get_features(self, mode='mlp'):
        # TODO: cnn, bilstm
        assert mode in ['mlp', 'cnn', 'bilstm'], "Classifier mode should be in ['mlp', 'cnn', 'bilstm']"
        self.encoder_hidden_size = self.encoder_output.shape[-1].value
        with tf.variable_scope("feature_layer"):
            if mode == 'mlp':
                mlp_weights = tf.get_variable("mlp_weights", [self.encoder_hidden_size, self.FLAGS.features_dim],
                                              initializer=tf.truncated_normal_initializer(stddev=0.02))
                mlp_bias = tf.get_variable("mlp_bias", [self.FLAGS.features_dim], initializer=tf.zeros_initializer())
                encoder_output_shape = tf.shape(self.encoder_output)
                # encoder_output_shape[-1] = self.FLAGS.features_dim
                features_shape = [encoder_output_shape[0], encoder_output_shape[1], self.FLAGS.features_dim]
                # features = tf.add(tf.matmul(self.encoder_output, mlp_weights), mlp_bias, name='features')
                features = tf.nn.xw_plus_b(tf.reshape(self.encoder_output, (-1, self.encoder_hidden_size)),
                                           mlp_weights, mlp_bias, name='features')
                features = tf.nn.relu(tf.reshape(features, shape=features_shape))
                features = tf.nn.dropout(features, keep_prob=self.dropout_keep_prob)
                features = tf.reduce_max(features, axis=1)
                return features  # (batch_size, hidden_size)

    def build_output_layer(self):
        with tf.variable_scope("output_layer"):
            # self.features.get_shape()[-1]
            classes_num = self.dm.label_num
            output_W = tf.get_variable('output_W', shape=[self.FLAGS.features_dim, classes_num], dtype=tf.float32,
                                       initializer=tf.truncated_normal_initializer(stddev=0.02))
            output_b = tf.get_variable('output_b', shape=[classes_num], dtype=tf.float32, initializer=tf.zeros_initializer())
            logits = tf.nn.xw_plus_b(self.features, output_W, output_b)
            return logits

    def compute_loss_acc(self):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
        self.predictions = tf.argmax(self.logits, axis=1, name='predictions')
        correct_predictions = tf.equal(self.predictions, self.y)
        self.probs = tf.nn.softmax(self.logits, name='predict_probability')
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

        if self.FLAGS.use_l2_loss:
            trainable_variables = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in trainable_variables if v.get_shape().ndims > 1])
            self.loss = self.loss + self.FLAGS.lambda_l2 * l2_loss

    def get_train_op(self, finetune_scope='all'):
        """
        :param finetune_scope: 'all', fine tune all layers;
                                'classifier', just train on classifier and fit encoder model;
                                'sbs', train on classifier and fine tune encoder model step by step backward;
                                'n', train on classifier and fine tune on the layer in self.encoder_layers[n:], fit the former (n-1) encoder layers
        :return: None
        """
        if self.FLAGS.optimizer_style == 'adam':
            optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate)
        elif self.FLAGS.optimizer_style == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.FLAGS.learning_rate)
        elif self.FLAGS.optimizer_style == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(self.FLAGS.learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.FLAGS.learning_rate)

        assert finetune_scope in ['all', 'classifier', 'step_by_step'] + [str(i) for i in range(len(self.encoder_layers))],\
                "Please make sure finetune_scope right!"

        self.trainable_variables = tf.trainable_variables()
        # print(self.trainable_variables)

        self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.trainable_variables), clip_norm=self.FLAGS.gradient_clip_val)
        self.train_op = optimizer.apply_gradients(zip(self.gradients, self.trainable_variables), global_step=self.global_step)

    def train(self):
        data = self.dm.get_data('train')
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(self.FLAGS.epoch_num):
            print("Epoch: {}".format(epoch))
            start_time = time.time()
            train_loss, train_acc = 0.0, 0.0
            preds = []
            random.shuffle(data)
            for current_batch_index, start in enumerate(range(0, len(data), self.FLAGS.batch_size)):
                batch_data = data[start:start+self.FLAGS.batch_size]
                input_ids, input_mask, segment_ids, label_ids = zip(*batch_data)
                feed_dict = {'input_ids:0': input_ids, 'y:0': label_ids, 'dropout_keep_prob:0': self.FLAGS.dropout_keep_prob}

                _, loss, acc, pred = self.sess.run(
                    [self.train_op, self.loss, self.accuracy, self.predictions], feed_dict=feed_dict)
                train_loss += loss
                train_acc += acc
                preds.extend(list(pred))

                if current_batch_index % self.FLAGS.batch_num_to_log == 0:
                    print(
                        "Epoch: {}, batch: {}, loss: {:.3f}, acc: {:.3f}".format(epoch, current_batch_index, loss, acc))


            #         # evaluate on validation set
            #         loss, accuracy, ps, top1, top3, top5 = evaluate(FLAGS, sess, valid_dm, valid_graph, 'search')
            #         print("Evaluate\tEpoch: {}, batch: {}, loss:{:.3f}, acc:{:.3f}".format(epoch, current_batch_index,
            #
            #         if top1 > best_top1:
            #             best_top1 = top1
            #             saver.save(sess, best_model_path)
            #             if os.path.exists(builder_saved_path):
            #                 import shutil
            #                 shutil.rmtree(builder_saved_path)
            #             builder = tf.saved_model.builder.SavedModelBuilder(builder_saved_path)
            #             builder.add_meta_graph_and_variables(sess, ['serve'])  # tf.local_variables_initializer())
            #             builder.save(True)
            #             print("save model done")
            #             # loss_test, accuracy_test, ps_test = evaluate(FLAGS, sess, test_dm, test_graph)
            #             # print(
            #             #     "Test\tEpoch: {}, batch: {}, loss:{:.3f}, acc:{:.3f}".format(epoch, current_batch_index,
            #             #                                                                      loss_test, accuracy_test))
            #
            # print("Epoch: {}, total_loss: {:.3f}, total_acc: {:.3f}, time: {:.3f}".format(epoch,
            #                                                                               float(
            #                                                                                   train_loss) / train_dm.num_batch_each_epoch,
            #                                                                               float(
            #                                                                                   train_acc) / train_dm.num_batch_each_epoch,
            #                                                                               time.time() - start_time))
            #
            # x1_parsed, x2_parsed, label = zip(*data)
            # target_names = ['class 0', 'class 1']
            # print("label2idx is: {}".format(train_dm.label2idx))
            # label = np.array(label)
            # preds = np.array(preds)
            # wrong = label[preds != label]
            # idx_1_wrong = np.sum(wrong) / len(wrong)
            # print("acc is: {}, idx 1 in wrong: {}\tidx 0 in wrong: {}".format(1 - len(wrong) / len(label), idx_1_wrong,
            #                                                                   1 - idx_1_wrong))
            # print(classification_report(label, preds, target_names=target_names))
            #

    def eval(self, texts_a=None, texts_b=None):
        if texts_a is not None:
            data = TextProcessor(self.FLAGS, texts_a, texts_b).data
        else:
            data = self.dm.get_data('eval')
        input_ids, input_mask, segment_ids, label_ids = zip(*data)
        feed_dict = {'input_ids:0': input_ids, 'dropout_keep_prob:0': 1.0, 'y:0': label_ids}
        loss, acc, pred = self.sess.run([self.loss, self.accuracy, self.predictions], feed_dict=feed_dict)
        print("loss: {:.3f}, acc: {:.3f}".format(loss, acc))
        return pred

    def predict(self, texts_a=None, texts_b=None, dropout_keep_prob=1.0):
        if texts_a is not None:
            data = TextProcessor(self.FLAGS, texts_a, texts_b).data
        else:
            data = self.dm.get_data('test')
        input_ids, input_mask, segment_ids, _ = zip(*data)
        feed_dict = {'input_ids:0': input_ids, 'dropout_keep_prob:0': dropout_keep_prob}
        preds, probs = self.sess.run([self.predictions, self.probs], feed_dict=feed_dict)
        return preds, probs
