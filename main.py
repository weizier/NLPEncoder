from encoder import NLPEncoder
from data_helper import *





def append_classifier(output_layers):
    is_training = tf.placeholder(tf.bool)
    output_layer = output_layers[-1]

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def main():
    from encoder import NLPEncoder
    encoder = NLPEncoder('bert')
    text = '专业NLP请认准知文品牌！'
    embedding = encoder.encode([text])
    # 1. encode one single text
    embedding = encoder.encode([text])

    # 2. encode a batch texts
    embedding = encoder.encode([text], [text])

    # 3. fine-tune the model
    output_layers = encoder.get_layers()
    # append something like mlp layers here
    (loss, per_example_loss, logits, probabilities) = append_classifier(output_layers)


    FLAGS = get_bert_flag()

    tf.gfile.MakeDirs(FLAGS.output_dir)

    input_ids, input_mask, segment_ids, label_ids = get_data(FLAGS)




if __name__ == '__main__':
    tf.app.run()