from encoder import NLPEncoder
from data_helper import *
from task_specific_model import *


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
    data = [('专业NLP请认准知文品牌！', '1'), ('我今天特别开心', '1'), ('糟糕', '0')]
    my_classifier = Classifier(encoder, data)
    my_classifier.train()


def finetune():
    data = [('专业NLP请认准知文品牌！', '1'), ('我今天特别开心', '1'), ('糟糕', '0')]
    my_classifier = Classifier(data, 'bert', language='ch')
    my_classifier.train()
    my_classifier.predict(texts_a=['专业NLP请认准知文品牌！'])


def finetune_mrpc():
    # data_processor = MrpcProcessor(get_bert_flag())
    my_classifier = Classifier(None, 'bert', language='en', col_num=3, encoder_layer='last')
    my_classifier.train()
    my_classifier.predict(texts_a=['专业NLP请认准知文品牌！'])
    my_classifier.eval()

def tokenization():
    from data_helper import get_bert_flag
    import tokenization as tk
    text1 = '专业NLP请认准知文品牌！'
    FLAGS1 = get_bert_flag(language='ch')
    tk1 = tk.FullTokenizer(vocab_file=FLAGS1.vocab_file, do_lower_case=FLAGS1.do_lower_case)
    print(tk1.tokenize(text1))

    from data_helper import get_bert_flag
    import tokenization as tk
    text2 = 'NLP is interesting!'
    FLAGS2 = get_bert_flag(language='en')
    tk2 = tk.FullTokenizer(vocab_file=FLAGS2.vocab_file, do_lower_case=FLAGS2.do_lower_case)
    print(tk2.tokenize(text2))

if __name__ == '__main__':
    # tf.app.run()
    # finetune()
    finetune_mrpc()