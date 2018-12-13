import os
import tensorflow as tf
import optimization
import tokenization
import csv
import random
import pickle

def get_bert_flag(language='en'):
    flags = tf.app.flags

    FLAGS = flags.FLAGS

    ## Required parameters

    if language == 'en':
        flags.DEFINE_string(
            "init_checkpoint", "./model/cased_L-12_H-768_A-12/bert_model.ckpt",
            "Initial checkpoint (usually from a pre-trained BERT model).")

        flags.DEFINE_string(
            "bert_config_file", "./model/cased_L-12_H-768_A-12/bert_config.json",
            "The config json file corresponding to the pre-trained BERT model. "
            "This specifies the model architecture.")

        flags.DEFINE_string("vocab_file", "./model/cased_L-12_H-768_A-12/vocab.txt",
                            "The vocabulary file that the BERT model was trained on.")

    else:
        flags.DEFINE_string(
            "init_checkpoint", "./model/chinese_L-12_H-768_A-12/bert_model.ckpt",
            "Initial checkpoint (usually from a pre-trained BERT model).")

        flags.DEFINE_string(
            "bert_config_file", "./model/chinese_L-12_H-768_A-12/bert_config.json",
            "The config json file corresponding to the pre-trained BERT model. "
            "This specifies the model architecture.")

        flags.DEFINE_string("vocab_file", "./model/chinese_L-12_H-768_A-12/vocab.txt",
                            "The vocabulary file that the BERT model was trained on.")

    # Other parameters
    # flags.DEFINE_string("task_name", 'mrpc', "The name of the task to train.")
    flags.DEFINE_string("task_name", 'qiqc_no_init', "The name of the task to train.")

    flags.DEFINE_string(
        "output_dir", './saved_model/',
        "The output directory where the model checkpoints will be written.")

    # flags.DEFINE_string(
    #     "data_dir", './data/glue_data/MRPC',
    #     "The input data dir. Should contain the .tsv files (or other data files) "
    #     "for the task.")
    flags.DEFINE_string(
        "data_dir", './data/QIQC',
        "The input data dir. Should contain the .tsv files (or other data files) "
        "for the task.")

    flags.DEFINE_bool(
        "do_lower_case", True,
        "Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")

    flags.DEFINE_integer(
        "max_seq_length", 128,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded.")

    flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
    return flags, FLAGS


def get_classifier_flag():
    flags, _ = get_bert_flag()

    # from bert
    flags.DEFINE_bool("do_train", False, "Whether to run training.")

    flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

    flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

    flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

    flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

    flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

    # flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

    flags.DEFINE_float("num_train_epochs", 3.0,
                       "Total number of training epochs to perform.")

    flags.DEFINE_float(
        "warmup_proportion", 0.1,
        "Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10% of training.")

    flags.DEFINE_integer("save_checkpoints_steps", 1000,
                         "How often to save the model checkpoint.")

    flags.DEFINE_integer("iterations_per_loop", 1000,
                         "How many steps to make in each estimator call.")

    # path
    flags.DEFINE_string("training_data_path", "./data/bank_of_china/toy.txt", "Default training data path.")
    flags.DEFINE_string("validation_data_path", "./data/bank_of_china/toy.txt", "Default validation data path.")
    flags.DEFINE_string("test_data_path", "./data/bank_of_china/toy.txt", "Default test data path.")
    flags.DEFINE_string("log_path", "./log/", "Default log path.")
    flags.DEFINE_string("word2vec_path", "./data/bank_of_china/data_corpus_fasttext_word.vec",
                        "Default word2vec data path.")
    flags.DEFINE_string("char2vec_path", "./data/bank_of_china/data_corpus_fasttext_char.vec",
                        "Default char2vec data path.")

    # task and dataset
    flags.DEFINE_string("language", "chinese", "Default language.")
    flags.DEFINE_string("task", "boa", "Default task.")
    flags.DEFINE_string("dataset", "boa", "Default dataset.")

    # data processing
    flags.DEFINE_boolean("data_augment", False, "Data augment or not? (default: True)")
    flags.DEFINE_boolean("balance_data", False, "Data balance or not? (default: True)")
    flags.DEFINE_boolean("use_parsing", False, "Use parser or not? (default: True)")

    # train

    flags.DEFINE_string("visible_gpus", "1", "visible gpus.")
    flags.DEFINE_float("learning_rate", 3e-5, "Learning rate. (default: 0.001)")
    flags.DEFINE_float("dropout_keep_prob", 0.9, "Dropout keep probability (default: 0.5)")
    flags.DEFINE_boolean("use_bn", False, "batch normalization? (default: False)")
    flags.DEFINE_integer("epoch_num", 10, "epoch number for training")
    flags.DEFINE_integer("epoch_cache_num", 3, "epoch number for training process on cache embedding to accelerate.")
    flags.DEFINE_integer("batch_size", 32, "batch size for training")
    flags.DEFINE_integer("batch_num_to_log", 5000, "batch number to print loss")
    flags.DEFINE_boolean("batch_with_same_length", False, "group training data with similar length? (default: True)")
    flags.DEFINE_boolean("use_random_valid", False, "random valid selection? (default: False)")

    # embedding layer
    flags.DEFINE_float("random_embedding_scale", 1.0, "random embedding scale (default: 1.0)")
    # word
    # flags.DEFINE_integer("max_seq_length", 50, "Max length of each sentence. (default: 50)")
    flags.DEFINE_boolean("use_random_word_embedding", False, "use random initialized word embedding? (default: False)")
    flags.DEFINE_integer("word_embedding_dim", 128, "Dimensionality of random word embedding (default: 128)")
    flags.DEFINE_boolean("dropout_word_embedding", True, "Dropout word embedding or not? (default: True)")
    flags.DEFINE_boolean("word_embedding_trainable", True, "Word embedding trainable or not? (default: True)")
    # char
    flags.DEFINE_integer("max_word_length", 4, "Max length of each word. (default: 10)")
    flags.DEFINE_boolean("use_char_embed", True, "Use char or not? (default: True)")
    flags.DEFINE_boolean("use_char_in_word", True, "Use char in word or across word? (default: True)")
    flags.DEFINE_boolean("use_random_char_embedding", False, "use random initialized char embedding? (default: False)")
    flags.DEFINE_integer("char_embedding_dim", 20, "Dimensionality of character embedding (default: 8)")
    flags.DEFINE_string("char_conv_out_channels", "100", "Char convolutional layer out channels.")
    flags.DEFINE_string("char_conv_windows", "2", "Char convolutional layer filter windows.")
    # pos
    flags.DEFINE_boolean("use_pos_embed", False, "Use pos or not? (default: True)")
    flags.DEFINE_boolean("use_pos_network_embed", False, "Use pos network or not? (default: True)")
    flags.DEFINE_integer("tag_num", 59, "pos tag number for one hot encoding")
    # other embedding way
    flags.DEFINE_boolean("use_self_att_embed", False, "Use self attention in embedding layer or not? (default: True)")
    flags.DEFINE_boolean("use_gcn_embed", False, "Use gcn in embedding layer or not? (default: True)")

    # encoder layer
    flags.DEFINE_boolean("use_highway_encoder", True, "Use highway layer or not? (default: True)")
    flags.DEFINE_integer("highway_layers_num", 5, "Highway_layers_num. (default: 1)")
    flags.DEFINE_integer("highway_output_size", 300, "Highway output size. (default: 0, same with highway's input)")
    flags.DEFINE_boolean("use_self_att_encoder", True, "Use self attention in encoder layer or not? (default: True)")
    flags.DEFINE_boolean("use_gcn_encoder", False, "Use graph convo net in encoder layer or not? (default: True)")
    flags.DEFINE_integer("gcn_layers_num", 1, "GCN layers num. (default: 1)")
    flags.DEFINE_integer("gcn_output_size", 300, "GCN output size. (default: 0, same with gcn's input)")
    flags.DEFINE_boolean("use_bilstm_encoder", True, "Use bilstm in encoder layer or not? (default: True)")
    flags.DEFINE_integer("bilstm_output_size", 300, "BiLSTM output size. (default: 0, same with its input)")
    flags.DEFINE_integer("bilstm_stack_num", 5, "BiLSTM stack layers num. (default: 1)")
    flags.DEFINE_string("encoded_layer_mixed_way", "concat", "Encoder mixed way.")

    # interaction layer
    flags.DEFINE_boolean("use_gcn_interaction", True, "Use gcn interaction layer? (default: True)")
    flags.DEFINE_boolean("use_QUAT_interaction", True, "Use QUAT in gcn interaction layer? (default: True)")
    flags.DEFINE_boolean("use_AUQT_interaction", True, "Use AUQT in gcn interaction layer? (default: True)")
    flags.DEFINE_integer("gcn_interaction_topk", 3, "GCN word-level interactive graph nodes num. (default: 1)")
    flags.DEFINE_boolean("use_negative_topk", False, "Use negative max value in gcn interaction layer? (default: True)")
    flags.DEFINE_boolean("use_gcn_self_interaction", False, "Use gcn self interaction layer? (default: True)")
    flags.DEFINE_string("interaction_mixed_way", "concat", "Different interaction methods mixed way.")

    # matching layer
    # flags.DEFINE_boolean("use_full_connect_matching", True, "Use full-connect in matching layer? (default: True)")
    flags.DEFINE_boolean("use_max_pooling_matching", False, "Use max pooling in matching layer? (default: True)")
    flags.DEFINE_boolean("use_attentive_pooling_matching", False, "Use attentive matching? (default: True)")
    flags.DEFINE_boolean("use_dot_attention", True, "use_dot_attention? (default: True)")
    flags.DEFINE_boolean("global_average_pooling_directly", True, "global_average_pooling_directly? (default: True)")
    # dense net
    flags.DEFINE_boolean("use_dense_matching", False, "Use dense network in matching layer or not? (default: True)")
    flags.DEFINE_boolean("first_scale_down_layer_relu", True, "first_scale_down_layer_relu. (default: True)")
    flags.DEFINE_float("dense_net_first_scale_down_ratio", 0.3, "dense_net_first_scale_down_ratio. (default: 0.3)")
    flags.DEFINE_float("dense_net_transition_rate", 0.5, "dense_net_transition_rate. (default: 0.5)")
    flags.DEFINE_integer("first_scale_down_kernel", 1, "first_scale_down_kernel. (default: 1)")
    flags.DEFINE_integer("dense_net_growth_rate", 20, "dense_net_growth_rate. (default: 20)")
    flags.DEFINE_integer("dense_net_layers", 8, "dense_net_layers. (default: 8)")
    flags.DEFINE_integer("dense_net_kernel_size", 3, "dense_net_kernel_size. (default: 3)")

    # self attention parameters
    flags.DEFINE_integer("self_attention_num_layer", 5, "self attention embedding layer number. (default: 1)")
    flags.DEFINE_integer("self_attention_key_dimension", 64, "self attention key projection dimension. (default: 64)")
    flags.DEFINE_integer("self_attention_value_dimension", 64,
                         "self attention value projection dimension. (default: 64)")
    flags.DEFINE_integer("self_attention_num_heads", 8, "self attention number of heads. (default: 8)")
    flags.DEFINE_boolean("is_residual", True, "whether use residual in self attention layer. (default: True)")
    flags.DEFINE_boolean("causality", False, "whether use future mask in self attention layer. (default: False)")
    flags.DEFINE_integer("feed_forward_dim", 256, "self attention feed forward dimension. (default: 2048)")

    # positional encoding parameter
    flags.DEFINE_boolean("use_position_encoding", False, "whether use position encoding. (default: True)")

    # features_layer
    flags.DEFINE_integer("features_dim", 796, "Features dimension. (default: 796)")

    # l2
    flags.DEFINE_boolean("use_l2_loss", False, "Use L2 loss or not? (default: True)")
    flags.DEFINE_float("lambda_l2", 0.00001, "l2 loss. (default: 0.0)")

    # optimizer
    flags.DEFINE_string("optimizer_style", "adam", "Train optimizer style.(default: adam)")
    flags.DEFINE_float("gradient_clip_val", 1.0, "Gradient clip value. (default: 1.0)")

    # device parameters
    flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    # FLAGS = flags._FlagValues()
    FLAGS = flags.FLAGS

    return FLAGS


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id)
  return feature


def convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to a TFRecord file."""

    input_ids, input_mask, segment_ids, label_ids = [], [], [], []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Processing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer) # InputFeatures instances

        input_ids.append(feature.input_ids)
        input_mask.append(feature.input_mask)
        segment_ids.append(feature.segment_ids)
        label_ids.append(feature.label_id)

    return input_ids, input_mask, segment_ids, label_ids


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, delimiter="\t", quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class TextProcessor:
    def __init__(self, FLAGS, texts_a, texts_b=None, labels=None):
        self.labels = labels
        self.label_list = self.get_label_list()
        self.data = self.get_processed_ids(FLAGS, texts_a, texts_b, labels)
        self.label_num = self.get_label_num()

    def get_label_num(self):
        return len(self.label_list)

    def get_labels(self):
        return self.labels

    def get_data(self, mode='train'):
        return self.data

    def get_label_list(self):
        if self.labels is None:
            return ['no_need']
        return sorted(list(set(self.labels)))

    def get_processed_ids(self, FLAGS, texts_a, texts_b=None, labels=None):
        """Creates examples for the training and dev sets."""
        assert (texts_b is None or len(texts_a) == len(texts_b)), "Number of two sequences must be the same! "
        # input_ids, input_mask, segment_ids, label_ids = [], [], [], []
        result = []
        tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
        for (i, text_a) in enumerate(texts_a):
            if i % 10000 == 0:
                tf.logging.info("Processing example %d." % i)
            guid = "%s-%s" % ("encode", i)
            text_a = tokenization.convert_to_unicode(text_a)
            label = self.label_list[0] if labels is None else labels[i]
            text_b = None
            if texts_b is not None:
                text_b = tokenization.convert_to_unicode(texts_b[i])

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

            feature = convert_single_example(i, example, self.label_list, FLAGS.max_seq_length, tokenizer)

            # input_ids.append(feature.input_ids)
            # input_mask.append(feature.input_mask)
            # segment_ids.append(feature.segment_ids)
            # label_ids.append(feature.label_id)
            result.append((feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id))

        return result


class ClassifierProcessor(DataProcessor):

    def __init__(self, FLAGS, filepath, col_num=2):
        self.FLAGS = FLAGS
        examples = self.get_train_examples(filepath, col_num)
        self.label_list = self.get_label_list()
        self.data = self.get_train_data(examples)

    def get_labels(self):
        return self.labels

    def get_label_list(self):
        if self.labels is None:
            return ['no_need']
        return sorted(list(set(self.labels)))

    def get_train_examples(self, filepath, col_num=2):
        """See base class."""
        lines = self._read_tsv(filepath)
        self.labels, examples = [], []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%d" % (i)
            assert len(line) == col_num, "Real column number is not equal with col_num on %d!" % i
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = None if col_num == 2 else tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[-1])
            self.labels.append(label)
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

    def get_train_data(self, examples):
        tokenizer = tokenization.FullTokenizer(vocab_file=self.FLAGS.vocab_file, do_lower_case=self.FLAGS.do_lower_case)
        input_ids, input_mask, segment_ids, label_ids = [], [], [], []
        for i, example in enumerate(examples):
            feature = convert_single_example(i, example, self.label_list, self.FLAGS.max_seq_length, tokenizer)

            input_ids.append(feature.input_ids)
            input_mask.append(feature.input_mask)
            segment_ids.append(feature.segment_ids)
            label_ids.append(feature.label_id)

        return input_ids, input_mask, segment_ids, label_ids


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "test_matched.tsv")),
      "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.label_list = self.get_labels()
        self.data_dir = self.FLAGS.data_dir
        self.label_num = 2

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[0])
            examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_data(self, data_style='train'):
        assert data_style in ['train', 'eval', 'test'], "data_style must be train, eval or test"
        if data_style == 'train':
            examples = self.get_train_examples(self.data_dir)
        elif data_style == 'eval':
            examples = self.get_dev_examples(self.data_dir)
        else:
            examples = self.get_test_examples(self.data_dir)
        tokenizer = tokenization.FullTokenizer(vocab_file=self.FLAGS.vocab_file, do_lower_case=self.FLAGS.do_lower_case)
        # input_ids, input_mask, segment_ids, label_ids = [], [], [], []
        result = []
        for i, example in enumerate(examples):
            feature = convert_single_example(i, example, self.label_list, self.FLAGS.max_seq_length, tokenizer)
            result.append((feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id))
        return result


class QiqcProcessor(DataProcessor):  # Quora Insincere Questions Classification
    """Processor for the QIQC data set."""
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.label_list = self.get_labels()
        self.data_dir = self.FLAGS.data_dir
        self.label_num = 2
        self.train_data = self.get_train_examples(self.data_dir)
        # random.shuffle(self.train_data)

    # def _read_tsv(self, input_file, delimiter=",", quotechar=None):
    #     """Reads a tab separated value file."""
    #     with open(input_file, "r", encoding='utf8') as f:
    #         reader = csv.reader(f, delimiter=",", quotechar=quotechar)
    #         lines = []
    #         for line in reader:
    #             lines.append(line)
    #         return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        data_cached_path = data_dir + '/train.cache'
        try:
            data = pickle.load(open(data_cached_path, 'rb'))
            print("Load train data from cache successfully.")
        except:
            path = os.path.join(data_dir, "train.csv")
            print("The train path is: %s" % path)
            data = self._create_examples(self._read_tsv(path, ',', '"'), "train")
            pickle.dump(data, open(data_cached_path, "wb"))
            print("Dump train data to cache successfully.")
        return data

    # def get_dev_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #     self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        data_cached_path = data_dir + '/test.cache'
        try:
            data = pickle.load(open(data_cached_path, 'rb'))
            print("Load test data from cache successfully.")
        except:
            path = os.path.join(data_dir, "test.csv")
            print("The test path is: %s" % path)
            data = self._create_examples(self._read_tsv(path, ',', '"'), "test")
            pickle.dump(data, open(data_cached_path, "wb"))
            print("Dump test data to cache successfully.")
        return data


    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                if len(line) != 2:
                    print("Something wrong happens at the %dth row!" % i)
                    continue
                label = "0"
            else:
                if len(line) != 3:
                    print("Something wrong happens at the %dth row!" % i)
                    continue
                label = tokenization.convert_to_unicode(line[2])
            text_a = tokenization.convert_to_unicode(line[1])
            # text_b = tokenization.convert_to_unicode(line[4])
            examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_data(self, data_style='train', split=0.9):
        assert data_style in ['train', 'eval', 'test'], "data_style must be train, eval or test"
        split_index = int(split * len(self.train_data))
        if data_style == 'train':
            examples = self.train_data[:split_index]
        elif data_style == 'eval':
            examples = self.train_data[split_index:]
        else:
            examples = self.get_test_examples(self.data_dir)

        data_cached_path = self.data_dir + '/{}_processed.cache'.format(data_style)
        try:
            result = pickle.load(open(data_cached_path, 'rb'))
            print("Load {}_processed data from cache successfully.".format(data_style))
        except:
            print("Now begin to process raw data...")
            tokenizer = tokenization.FullTokenizer(vocab_file=self.FLAGS.vocab_file, do_lower_case=self.FLAGS.do_lower_case)
            result = []
            for i, example in enumerate(examples):
                feature = convert_single_example(i, example, self.label_list, self.FLAGS.max_seq_length, tokenizer)
                result.append((feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id))
            pickle.dump(result, open(data_cached_path, "wb"))
            print("Dump {}_processed data to cache successfully.".format(data_style))
        return result

    def balance_data(self, data):
        print("Begin to up sampling postive data...")
        pos_data, neg_data = [], []
        for d in data:
            if d[-1]:
                pos_data.append(d)
            else:
                neg_data.append(d)
        upsample = int(len(neg_data) / len(pos_data))
        print('up sampling to {}'.format(upsample))
        return pos_data * upsample + neg_data


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def get_data(FLAGS):

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "xnli": XnliProcessor,
    }
    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)  # InputExample instances
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        # input_ids, input_mask, segment_ids, label_ids = \
        return convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer)


def build_data_processor(FLAGS, path_or_data, col_num):
    # assert isinstance(path_or_data, (str, list, MnliProcessor, ColaProcessor, MrpcProcessor, ClassifierProcessor,
    #                                  XnliProcessor)), "Make sure path_or_data is a filepath or list of data!"
    assert isinstance(path_or_data, (str, list)), "Make sure path_or_data is a filepath or list of data!"
    if path_or_data in ['mnli', 'cola', 'mrpc', 'xnli', 'qiqc']:
        processors = {
            "cola": ColaProcessor,
            "mnli": MnliProcessor,
            "mrpc": MrpcProcessor,
            "xnli": XnliProcessor,
            "qiqc": QiqcProcessor
        }
        return processors[path_or_data](FLAGS)
    if isinstance(path_or_data, str):
        return ClassifierProcessor(FLAGS, path_or_data, col_num)
    if isinstance(path_or_data, list):
        if col_num == 2:
            texts_a, labels = zip(*path_or_data)
            texts_b = None
        elif col_num == 3:
            texts_a, texts_b, labels = zip(*path_or_data)
        else:
            raise ValueError
        return TextProcessor(FLAGS, texts_a, texts_b, labels)
