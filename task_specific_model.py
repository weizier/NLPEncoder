import tensorflow as tf
from data_helper import *
import time
import random
from encoder import NLPEncoder
from tensorflow.python.client import timeline
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np

# TODO: Squad, seq2seq, crf, qa,

class TaskSpecificModel:
    def __init__(self, encoder):
        pass

    def get_encoder_output(self):
        raise NotImplementedError("You must implement encoder function.")

    def get_output_layer(self):
        raise NotImplementedError("You must implement output function.")


class Classifier(TaskSpecificModel):
    """
    TODO: finetuning on 1M rows of data take 24 hours, two ways to solve it:
        1. finetune all layers some time, then cache the output of bert and just finetune classifier layer
        2. try multiple gpus
    """

    def __init__(self, encoder='bert', language='en', comebine_encoder_mode='cls', feature_mode='max', finetune_scope='all', path_or_data='mrpc', col_num=3, init_from_check=True):
        self.FLAGS = get_classifier_flag()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.FLAGS.visible_gpus
        self.encoder = self.get_encoder(encoder, language, init_from_check)
        self.dropout_keep_prob = self.encoder.model.dropout_keep_prob
        self.sess = self.get_session()
        self.dm = build_data_processor(self.FLAGS, path_or_data, col_num)  # you can change the data manager when training
        self.create_placeholders()

        with tf.variable_scope("classifier"):
            self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
            self.encoder_output = self.get_encoder_output(comebine_encoder_mode=comebine_encoder_mode)
            self.features = self.get_features(feature_mode)
            self.logits = self.build_output_layer()
            self.compute_loss_acc()
            self.get_train_op(finetune_scope)

        print("Graph created successfully, now begin to initialize all variables.")
        self.sess.run(tf.global_variables_initializer())
        print("Encoder and Classifier variables initialized successfully!")
        self.build_saved_path(comebine_encoder_mode, feature_mode, finetune_scope)
        self.is_restored = False

    def get_encoder(self, encoder='bert', language='en', init_from_check=True):
        return NLPEncoder(FLAGS=self.FLAGS, model_name=encoder, language=language, init_from_check=init_from_check)

    def get_session(self):
        config = tf.ConfigProto(graph_options=tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        sess = tf.Session(config=config)
        # self.sess = tf.Session()
        # data_processor = MrpcProcessor(self.encoder.FLAGS)
        # path_or_data = data_processor
        return sess

    def create_placeholders(self):
        self.y = tf.placeholder(dtype=tf.int64, shape=None, name="y")

    def get_encoder_output(self, comebine_encoder_mode='cls'):
        """
        :param mode: 'attention' is a softmax weight on all of the layers;
                      'last' just use the last layer of encoder(but 'cls' token in bert)
                      'let_me_do': a list of all layers in encoder will return, combination will let you make it.
        :return: a combination of all encoders or a list of all layers in encoder
        """
        self.encoder_layers = self.encoder.get_layers()

        assert comebine_encoder_mode in ['attention', 'cls', 'last', 'let_me_do'], "Specify a mode in ['attention', 'last', 'let_me_do']"
        if comebine_encoder_mode == 'let_me_do':
            return self.encoder_layers   # [13 * (batch_size, seq_length, hidden_size) + (batch_size, hidden_size)]
        elif comebine_encoder_mode == 'cls':
            return tf.expand_dims(self.encoder_layers[-1], 1, name='encoder_output')  # (batch_size, 1, hidden_size)
        elif comebine_encoder_mode == 'last':
            return self.encoder_layers[-2]  # (batch_size, seq_length, hidden_size)

        with tf.variable_scope("attention_combine"):
            hidden_size = self.encoder_layers[0].shape[-1].value

            combine_weights = tf.get_variable(
                "attention_combine_weights", [hidden_size, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            # combine_bias = tf.get_variable(
            #     "attention_combine_bias", [hidden_size], initializer=tf.zeros_initializer())
            combine_v = tf.get_variable(
                "attention_combine_v", [hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
            # [13 * (batch_size, seq_length, hidden_size)] --> (batch_size, seq_length, 13, hidden_size)
            self.combine_in = tf.concat([tf.expand_dims(layer, axis=2) for layer in self.encoder_layers[:-1]], axis=2)
            combine_in_shape = tf.shape(self.combine_in)# self.combine_in.shape
            self.combine_e = tf.nn.tanh(tf.matmul(tf.reshape(self.combine_in, [-1, hidden_size]), combine_weights))
            self.combine_e = tf.reshape(self.combine_e, combine_in_shape)  # (batch_size, seq_length, 13, hidden_size)
            self.combine_e = tf.reduce_sum(combine_v * self.combine_e, -1)  # (batch_size, seq_length, 13)
            # self.combine_e = tf.nn.dropout(self.combine_e, keep_prob=self.dropout_keep_prob)
            # (batch_size, seq_length, 13, hidden_size) --> (batch_size, seq_length, hidden_size)
            encoder_output = tf.squeeze(tf.matmul(tf.expand_dims(tf.nn.softmax(self.combine_e), 2), self.combine_in), 2)
            encoder_output = tf.nn.dropout(encoder_output, keep_prob=self.dropout_keep_prob)
            return encoder_output  # (batch_size, seq_length, hidden_size)

    def get_features(self, mode='max'):
        # TODO: cnn, bilstm
        assert mode in ['max', 'mlp', 'cnn', 'bilstm', 'attention'], "Classifier mode should be in ['mlp', 'cnn', 'bilstm']"
        self.encoder_hidden_size = self.encoder_output.shape[-1].value
        encoder_output_shape = tf.shape(self.encoder_output)
        with tf.variable_scope("feature_layer"):
            if mode == 'mlp':
                mlp_weights = tf.get_variable("mlp_weights", [self.encoder_hidden_size, self.FLAGS.features_dim],
                                              initializer=tf.truncated_normal_initializer(stddev=0.02))
                mlp_bias = tf.get_variable("mlp_bias", [self.FLAGS.features_dim], initializer=tf.zeros_initializer())
                # encoder_output_shape[-1] = self.FLAGS.features_dim
                features_shape = [encoder_output_shape[0], encoder_output_shape[1], self.FLAGS.features_dim]
                # features = tf.add(tf.matmul(self.encoder_output, mlp_weights), mlp_bias, name='features')
                features = tf.nn.xw_plus_b(tf.reshape(self.encoder_output, (-1, self.encoder_hidden_size)),
                                           mlp_weights, mlp_bias, name='features')
                # (batch_size, seq_length, features_dim)
                features = tf.nn.relu(tf.reshape(features, shape=features_shape))
                features = tf.nn.dropout(features, keep_prob=self.dropout_keep_prob)
                features = tf.reduce_max(features, axis=1)
                return features  # (batch_size, features_dim)
            elif mode == 'max':
                return tf.reduce_max(self.encoder_output, axis=1)
            elif mode == 'attention':
                attention_w = tf.get_variable('feature_attention_w', [self.encoder_hidden_size, self.FLAGS.features_dim],
                                              initializer=tf.truncated_normal_initializer(stddev=0.02))
                attention_v = tf.get_variable('feature_attention_v', [self.FLAGS.features_dim],
                                              initializer=tf.truncated_normal_initializer(stddev=0.02))

                # (batch_size * seq_len, hidden_size) --> (batch_size * seq_len, features_dim)
                e = tf.nn.tanh(tf.matmul(tf.reshape(self.encoder_output, (-1, self.encoder_hidden_size)), attention_w))

                # (batch_size * seq_len, features_dim) --> (batch_size * seq_len) --> (batch_size, seq_len)
                e = tf.reshape(tf.reduce_sum(attention_v * e, -1), [encoder_output_shape[0], encoder_output_shape[1]])
                alpha = tf.nn.softmax(e)  # (batch_size, seq_len)

                features = tf.squeeze(tf.matmul(tf.expand_dims(alpha, 1), self.encoder_output), 1)

                return features

    def build_output_layer(self):
        with tf.variable_scope("output_layer"):
            features_dim = self.features.get_shape()[-1]
            classes_num = self.dm.label_num
            output_W = tf.get_variable('output_W', shape=[features_dim, classes_num], dtype=tf.float32,
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
            self.optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate)
        elif self.FLAGS.optimizer_style == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.FLAGS.learning_rate)
        elif self.FLAGS.optimizer_style == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.FLAGS.learning_rate)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(self.FLAGS.learning_rate)

        assert finetune_scope in ['all', 'classifier', 'step_by_step'] + [str(i) for i in range(len(self.encoder_layers))],\
                "Please make sure finetune_scope right!"

        self.trainable_variables = tf.trainable_variables()
        # tvars = []
        print("*******All trainable variables********")
        print(len(self.trainable_variables))
        print('\n'.join([str(var) for var in self.trainable_variables]))
        print('\n')
        if finetune_scope == 'classifier':
            tvars = [var for var in self.trainable_variables if 'bert' not in var.name]
            self.trainable_variables = tvars
            print("*******Classifier trainable variables********")
            print(len(self.trainable_variables))
            print('\n'.join([str(var) for var in self.trainable_variables]))
            print('\n')

        print("Now, begin to compute gradients on trainable variables...")
        self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.trainable_variables), clip_norm=self.FLAGS.gradient_clip_val)
        self.train_op = self.optimizer.apply_gradients(zip(self.gradients, self.trainable_variables), global_step=self.global_step)
        print("Computing gradients on trainable variables has done!")

    def get_train_op_under_cache(self):
        print("*******Trainable variables under cache********")
        # print(len(self.trainable_variables))
        # print('\n'.join([str(var) for var in self.trainable_variables]))
        # print('\n')
        self.trainable_variables = [var for var in tf.trainable_variables() if var.name.startswith('classifier')]
        print(len(self.trainable_variables))
        print('\n'.join([str(var) for var in self.trainable_variables]))
        print('\n')

        print("Now, begin to compute gradients on trainable variables under cache...")
        self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.trainable_variables), clip_norm=self.FLAGS.gradient_clip_val)
        self.train_op = self.optimizer.apply_gradients(zip(self.gradients, self.trainable_variables), global_step=self.global_step)
        print("Computing gradients on trainable variables under cache has done!")

    def build_saved_path(self, comebine_encoder_mode, feature_mode, finetune_scope):
        self.saver_saved_dir = self.FLAGS.output_dir + "/{}/saver/".format(self.FLAGS.task_name)
        self.saver_saved_path = self.saver_saved_dir + "{}_{}_{}".format(comebine_encoder_mode, feature_mode, finetune_scope)
        self.builder_saved_dir = self.FLAGS.output_dir + "/{}/builder/".format(self.FLAGS.task_name)
        self.builder_saved_path = self.builder_saved_dir

    def restore_model(self, saver_saved_dir, saver_saved_path, saver=None):
        if saver is None:
            saver = tf.train.Saver()
        if os.path.exists(saver_saved_dir):
            print("restore model from {}".format(saver_saved_path))
            saver.restore(self.sess, saver_saved_path)
            self.is_restored = True
            print("restore model done.")
        else:
            raise FileNotFoundError("File not exists.")
        return saver

    def cache_nodes_value(self, finetune_scope, comebine_encoder_mode, data, path):
        # data = self.dm.get_data('train')
        if finetune_scope == 'classifier':
            for current_batch_index, start in enumerate(range(0, len(data), self.FLAGS.batch_size)):
                batch_data = data[start:start+self.FLAGS.batch_size]
                input_ids, input_mask, segment_ids, label_ids = zip(*batch_data)
                feed_dict = {'input_ids:0': input_ids, 'dropout_keep_prob:0': 1.0}

                # [13 * (batch_size, seq_length, hidden_size) + (batch_size, hidden_size)]
                encoder_layers_value = self.sess.run(self.encoder_layers, feed_dict=feed_dict)

                encoder_feed_dict = {}
                if comebine_encoder_mode == 'cls':
                    pickle.dump(encoder_layers_value[-1], open("{}/cache/{}_cls.cache".format(path, current_batch_index),'wb'))
                elif comebine_encoder_mode == 'last':
                    pickle.dump(encoder_layers_value[-2], open("{}/cache/{}_last.cache".format(path, current_batch_index),'wb'))
                elif comebine_encoder_mode == 'attention':
                    pickle.dump(encoder_layers_value[:-1], open("{}/cache/{}_att.cache".format(path, current_batch_index),'wb'))
                elif comebine_encoder_mode == 'let_me_do':
                    pickle.dump(encoder_layers_value, open("{}/cache/{}_lmd.cache".format(path, current_batch_index),'wb'))



            # output = self.sess.run()

    def train(self, use_cache=False, train_on_cache=False, cache_nodes=True, balance=True, train_all=True):
        data = self.dm.get_data('train')
        if balance:
            data = self.dm.balance_data(data)
        print("Train data load successfully! Train data has %d rows." % len(data))
        # with tf.Session() as sess:
        # self.sess.run(tf.global_variables_initializer())
        # print("*******Before restore before finetune, All trainable variables********")
        # print(len(tf.trainable_variables()))
        # print("\n\nBefore restore, before finetune, the variable is:")
        # a, b = self.sess.run(['bert/encoder/layer_11/output/dense/bias:0', 'classifier/output_layer/output_W:0'])
        # print(a[:10])
        # print(b[:10])
        # print('\n\n')

        # saver_saved_dir = self.FLAGS.output_dir + "/{}/saver/".format(self.FLAGS.task_name)
        # saver_saved_path = saver_saved_dir + 'best2'
        # builder_saved_dir = self.FLAGS.output_dir + "/{}/builder/".format(self.FLAGS.task_name)
        # builder_saved_path = builder_saved_dir
        # saver = tf.train.Saver()
        # if use_cache and os.path.exists(saver_saved_dir):
        #     print("restore model from {}".format(saver_saved_dir))
        #     saver.restore(self.sess, saver_saved_path)
        #     print("restore model done.")

        if use_cache:
            saver = self.restore_model(self.saver_saved_dir, self.saver_saved_path)
        else:
            saver = tf.train.Saver()

        # print("*******After restore before finetune, All trainable variables********")
        # print(len(tf.trainable_variables()))
        # print("\n\nAfter restore, before finetune, the variable is:")
        # a, b = self.sess.run(['bert/encoder/layer_11/output/dense/bias:0', 'classifier/output_layer/output_W:0'])
        # print(a[:10])
        # print(b[:10])
        # print('\n\n')

        best_acc = -1
        for epoch in range(self.FLAGS.epoch_num):
            print("Epoch: {}".format(epoch))
            start_time = time.time()
            train_loss, train_acc = 0.0, 0.0
            preds = []
            random.shuffle(data)
            # for current_batch_index, start in enumerate(range(0, len(data), self.FLAGS.batch_size)):
            for current_batch_index, start in enumerate(range(0, len(data), self.FLAGS.batch_size)):
                batch_data = data[start:start+self.FLAGS.batch_size]
                input_ids, input_mask, segment_ids, label_ids = zip(*batch_data)
                feed_dict = {'input_ids:0': input_ids, 'y:0': label_ids, 'dropout_keep_prob:0': self.FLAGS.dropout_keep_prob}

                # print("1111111111")
                # run_metadata = tf.RunMetadata()
                # print("2222222")
                # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # print("3333333")
                # _, loss, acc, pred = self.sess.run([self.train_op, self.loss, self.accuracy, self.predictions],
                #                                          options=run_options, run_metadata=run_metadata,
                #                                          feed_dict=feed_dict)
                _, loss, acc, pred = self.sess.run([self.train_op, self.loss, self.accuracy, self.predictions],
                                                         feed_dict=feed_dict)
                train_loss += loss * len(batch_data)
                train_acc += acc * len(batch_data)
                preds.extend(list(pred))

                if current_batch_index and current_batch_index % self.FLAGS.batch_num_to_log == 0:
                    print(
                        "Epoch: {}, batch: {}, loss: {:.3f}, acc: {:.3f}".format(epoch, current_batch_index, loss, acc))
                    # print("555555555")
                    # tl = timeline.Timeline(run_metadata.step_stats)
                    # ctf = tl.generate_chrome_trace_format()
                    # with open('timeline_{}.json'.format(current_batch_index), 'w') as wd:
                    #     wd.write(ctf)
                    # print("*******After111 restore after finetune, All trainable variables********")
                    # print(len(tf.trainable_variables()))
                    # print("\n\nAfter restore, after finetune, the variable is:")
                    # a, b = self.sess.run(
                    #     ['bert/encoder/layer_11/output/dense/bias:0', 'classifier/output_layer/output_W:0'])
                    # print(a[:10])
                    # print(b[:10])
                    # print('\n\n')

                    loss, acc, pred, target_f1 = self.eval()
                    if target_f1 > best_acc:
                        print("Save model to {}".format(self.saver_saved_path))
                        best_acc = target_f1
                        saver.save(self.sess, self.saver_saved_path)
                        if os.path.exists(self.builder_saved_dir):
                            import shutil
                            shutil.rmtree(self.builder_saved_dir)
                        builder = tf.saved_model.builder.SavedModelBuilder(self.builder_saved_path)
                        builder.add_meta_graph_and_variables(self.sess, ['serve'])  # tf.local_variables_initializer())
                        builder.save(True)
                        print("Save model done.")



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
            print("Train, loss: {:.3f}, acc: {:.3f}".format(train_loss/len(data), train_acc/len(data)))
            loss, acc, pred, target_f1 = self.eval()
            if target_f1 > best_acc:
                best_acc = target_f1
                saver.save(self.sess, self.saver_saved_path)
                if os.path.exists(self.builder_saved_dir):
                    import shutil
                    shutil.rmtree(self.builder_saved_dir)
                builder = tf.saved_model.builder.SavedModelBuilder(self.builder_saved_path)
                builder.add_meta_graph_and_variables(self.sess, ['serve'])  # tf.local_variables_initializer())
                builder.save(True)
                print("save model done")

        print("Finetuning encoder has done!")

        if train_on_cache:
            data = self.dm.get_data('train')
            try:
                self.restore_model(self.saver_saved_dir, self.saver_saved_path, saver)
            except:
                print("Restore from cache model failed!")
            if cache_nodes:
                print("Begin to cache encoder output...")
                self.cache_nodes_value('classifier', 'cls', data, self.FLAGS.data_dir)
            print("Cache encoder output done! Now, begin to train on cached embeddings...")
            self.get_train_op_under_cache()
            for epoch in range(self.FLAGS.epoch_cache_num):
                print("Epoch on cache: {}".format(epoch))
                train_loss, train_acc = 0.0, 0.0
                preds = []
                for current_batch_index, start in enumerate(range(0, len(data), self.FLAGS.batch_size)):
                    batch_data = data[start:start+self.FLAGS.batch_size]
                    input_ids, input_mask, segment_ids, label_ids = zip(*batch_data)
                    cache_path = "{}/cache/{}_cls.cache".format(self.FLAGS.data_dir, current_batch_index)
                    cls_embedding = pickle.load(open(cache_path, 'rb'))
                    # feed_dict = {self.encoder.model.pooled_output: cls_embedding, 'y:0': label_ids,
                    #              'dropout_keep_prob:0': self.FLAGS.dropout_keep_prob}
                    # feed_dict = {self.encoder_output: np.expand_dims(cls_embedding, 1), 'y:0': label_ids,
                    #              'dropout_keep_prob:0': self.FLAGS.dropout_keep_prob}
                    feed_dict = {'classifier/encoder_output:0': np.expand_dims(cls_embedding, 1),
                                 'y:0': label_ids,
                                 'dropout_keep_prob:0': self.FLAGS.dropout_keep_prob}

                    _, loss, acc, pred = self.sess.run([self.train_op, self.loss, self.accuracy, self.predictions],
                                                             feed_dict=feed_dict)
                    train_loss += loss * len(batch_data)
                    train_acc += acc * len(batch_data)
                    preds.extend(list(pred))

                    if current_batch_index % self.FLAGS.batch_num_to_log == 0:
                        print(
                            "Epoch: {}, batch: {}, loss: {:.3f}, acc: {:.3f}".format(epoch, current_batch_index, loss, acc))
                        loss, acc, pred = self.eval()
                        if acc > best_acc:
                            print("Save model to {}".format(self.saver_saved_path+'cache'))
                            best_acc = acc
                            saver.save(self.sess, self.saver_saved_path)
                            # if os.path.exists(self.builder_saved_dir):
                            #     import shutil
                            #     shutil.rmtree(self.builder_saved_dir)
                            # builder = tf.saved_model.builder.SavedModelBuilder(self.builder_saved_path)
                            # builder.add_meta_graph_and_variables(self.sess, ['serve'])  # tf.local_variables_initializer())
                            # builder.save(True)
                            print("Save model done.")

                print("Train, loss: {:.3f}, acc: {:.3f}".format(train_loss/len(data), train_acc/len(data)))
                loss, acc, pred = self.eval()
                if acc > best_acc:
                    best_acc = acc
                    saver.save(self.sess, self.saver_saved_path+'cache')
                    # if os.path.exists(self.builder_saved_dir):
                    #     import shutil
                    #     shutil.rmtree(self.builder_saved_dir)
                    # builder = tf.saved_model.builder.SavedModelBuilder(self.builder_saved_path)
                    # builder.add_meta_graph_and_variables(self.sess, ['serve'])  # tf.local_variables_initializer())
                    # builder.save(True)
                    print("save model done")

        if train_all:
            print("Begin to train the whole data...")
            # my_classifier.is_restored = False
            saver = self.restore_model(self.saver_saved_dir, self.saver_saved_path)
            data = data + self.dm.balance_data(self.dm.get_data('eval'))
            train_loss, train_acc = 0.0, 0.0
            preds = []
            random.shuffle(data)
            # for current_batch_index, start in enumerate(range(0, len(data), self.FLAGS.batch_size)):
            for current_batch_index, start in enumerate(range(0, len(data), self.FLAGS.batch_size)):
                batch_data = data[start:start + self.FLAGS.batch_size]
                input_ids, input_mask, segment_ids, label_ids = zip(*batch_data)
                feed_dict = {'input_ids:0': input_ids, 'y:0': label_ids,
                             'dropout_keep_prob:0': self.FLAGS.dropout_keep_prob}

                _, loss, acc, pred = self.sess.run([self.train_op, self.loss, self.accuracy, self.predictions],
                                                   feed_dict=feed_dict)
                train_loss += loss * len(batch_data)
                train_acc += acc * len(batch_data)
                preds.extend(list(pred))
                if current_batch_index and current_batch_index % self.FLAGS.batch_num_to_log == 0:
                    print("Batch: {}, loss: {:.3f}, acc: {:.3f}".format(current_batch_index, loss, acc))

            print("Save model to {}".format(self.saver_saved_path))
            saver.save(self.sess, self.saver_saved_path)
            if os.path.exists(self.builder_saved_dir):
                import shutil
                shutil.rmtree(self.builder_saved_dir)
            builder = tf.saved_model.builder.SavedModelBuilder(self.builder_saved_path)
            builder.add_meta_graph_and_variables(self.sess, ['serve'])  # tf.local_variables_initializer())
            builder.save(True)
            print("Save model done.")

    def eval(self, texts_a=None, texts_b=None):
        if texts_a is not None:
            data = TextProcessor(self.FLAGS, texts_a, texts_b).data
        else:
            data = self.dm.get_data('eval')
        eval_loss, eval_acc = 0.0, 0.0
        ground_ids, preds = [], []
        for current_batch_index, start in enumerate(range(0, len(data), self.FLAGS.batch_size)):
            batch_data = data[start:start + self.FLAGS.batch_size]
            input_ids, input_mask, segment_ids, label_ids = zip(*batch_data)
            feed_dict = {'input_ids:0': input_ids, 'dropout_keep_prob:0': 1.0, 'y:0': label_ids}
            loss, acc, pred = self.sess.run([self.loss, self.accuracy, self.predictions], feed_dict=feed_dict)
            eval_loss += loss * len(batch_data)
            eval_acc += acc * len(batch_data)
            preds.extend(list(pred))
            ground_ids.extend(label_ids)
        print("Evaluation, loss: {:.3f}, acc: {:.3f}".format(eval_loss/len(data), eval_acc/len(data)))
        target_names = ['class 0', 'class 1']
        print(classification_report(ground_ids, preds, target_names=target_names))
        p, r, f1, s = precision_recall_fscore_support(ground_ids, preds, labels=[1], average=None)
        print("Target precision: {}, recal: {}, f1:{}".format(p, r, f1))
        return eval_loss, eval_acc, preds, f1

    def predict(self, texts_a=None, texts_b=None, dropout_keep_prob=1.0):
        if texts_a is not None:
            data = TextProcessor(self.FLAGS, texts_a, texts_b).data
        else:
            data = self.dm.get_data('test')
        if not self.is_restored:
            self.restore_model(self.saver_saved_dir, self.saver_saved_path)
        preds, probs = [], []
        for current_batch_index, start in enumerate(range(0, len(data), self.FLAGS.batch_size)):
            batch_data = data[start:start + self.FLAGS.batch_size]
            input_ids, input_mask, segment_ids, label_ids = zip(*batch_data)
            feed_dict = {'input_ids:0': input_ids, 'dropout_keep_prob:0': dropout_keep_prob}
            pred, prob = self.sess.run([self.predictions, self.probs], feed_dict=feed_dict)
            preds.extend(list(pred))
            probs.extend(list(prob))
            if current_batch_index % self.FLAGS.batch_num_to_log == 0:
                print("current batch is: %d" % current_batch_index)
        with open("predict.result", 'w') as f:
            for a, b in zip(preds, probs):
                f.write("{}\t{}\n".format(a, b))
        return preds, probs
    # def test_builder(self):
    #     created_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    #     builder_saved_dir = self.FLAGS.output_dir + self.FLAGS.task_name
    #     # builder_saved_path = "{}/{}-saved-model".format(builder_saved_dir, created_time)
    #     builder_saved_path = builder_saved_dir
    #     print("builder saved path is: {}".format(builder_saved_path))
    #     if use_cache and os.path.exists(builder_saved_dir):
    #         print("restore model from {}".format(builder_saved_dir))
    #         # self.sess.reset()
    #         # self.sess = tf.Session(graph=tf.Graph())
    #         tf.saved_model.loader.load(self.sess, ['train'], builder_saved_dir)
    #         print("restore model done.")
    #
    #         print("Save model to {}".format(builder_saved_dir))
    #         best_acc = acc
    #         if os.path.exists(builder_saved_dir):
    #             import shutil
    #             shutil.rmtree(builder_saved_dir)
    #             print("Delete the builder_saved_dir done.")
    #         builder = tf.saved_model.builder.SavedModelBuilder(builder_saved_path)
    #         print("*******After2222 restore after finetune, All trainable variables********")
    #         print(len(tf.trainable_variables()))
    #         builder.add_meta_graph_and_variables(self.sess, ['train'])  # tf.local_variables_initializer())
    #         builder.save(True)
    #         print("save model done")

