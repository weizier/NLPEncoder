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
    data = [('专业NLP请认准知文品牌！', 'dfdf', '1'), ('我今天特别开心', 'dfdf', '1'), ('糟糕', 'fddfdsf', '0')]
    my_classifier = Classifier('bert', language='ch', comebine_encoder_mode='attention', path_or_data=data, col_num=3)
    my_classifier.train()
    my_classifier.predict(texts_a=['专业NLP请认准知文品牌！'])

def finetune_mrpc():
    my_classifier = Classifier('bert', language='en', comebine_encoder_mode='attention', finetune_scope='classifier', path_or_data='mrpc', col_num=3)
    my_classifier.train(True)
    my_classifier.predict(texts_a=['专业NLP请认准知文品牌！'])
    my_classifier.eval()

def finetune_qiqc():
    my_classifier = Classifier('bert', language='en', comebine_encoder_mode='attention', feature_mode='attention', finetune_scope='all', path_or_data='qiqc', col_num=2, init_from_check=False)
    my_classifier.train(balance=False)
    # my_classifier.predict(texts_a=['专业NLP请认准知文品牌！'])
    # my_classifier.eval()
    my_classifier.is_restored = False
    preds, probs = my_classifier.predict()
    res = [i for i, n in enumerate(preds) if n == 1]
    print(res)

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
    # finetune_mrpc()
    finetune_qiqc()