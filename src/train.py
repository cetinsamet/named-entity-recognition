import time
import gzip

from nltk.tag import pos_tag
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.externals import joblib


class NamedEntityRecognizer():
    """Named Entity Recognition class for the English language"""

    def __init__(self):
        self.sentences  = list()
        self.features   = list()
        self.ne_labels  = list()
        self.vectorizer = DictVectorizer()
        self.model      = LogisticRegressionCV(random_state=123)

    def read_data(self, train_datapath):
        """Read sentences from given corpus data"""
        self.sentences  = []
        with open(train_datapath, 'r') as infile:
            sent = []
            for line in infile:
                line = str.split(str.strip(line), '\t')
                if len(line) == 4:
                    token, pos_tag, ne_tag = line[0], line[1], line[3]
                    sent.append((token, pos_tag, ne_tag))
                    continue
                self.sentences.append(sent)
                sent = []
        print("-> %d sentences are read from '%s'." % (len(self.sentences), train_datapath))
        return

    def get_feature(self, token, token_index, sent, pos_tag):
        """Extract features of given word(token)"""
        token_feature = {
                        'token'             : token,
                        'is_first'          : token_index == 0,
                        'is_last'           : token_index == len(sent)-1,

                        'is_capitalized'    : token[0].upper() == token[0],
                        'is_all_capitalized': token.upper() == token,
                        'is_capitals_inside': token[1:].lower() != token[1:],
                        'is_numeric'        : token.isdigit(),

                        'prefix-1'          : token[0],
                        'prefix-2'          : '' if len(token) < 2  else token[:1],

                        'suffix-1'          : token[-1],
                        'suffix-2'          : '' if len(token) < 2  else token[-2:],

                        'prev-token'        : '' if token_index == 0     else sent[token_index - 1][0],
                        '2-prev-token'      : '' if token_index <= 1     else sent[token_index - 2][0],

                        'next-token'        : '' if token_index == len(sent) - 1     else sent[token_index + 1][0],
                        '2-next-token'      : '' if token_index >= len(sent) - 2     else sent[token_index + 2][0],
                        'pos-tag'           : pos_tag
                        }
        return  token_feature

    def form_data(self):
        """Create datasets for training/evaluation/testing"""
        self.features   = []
        self.ne_labels  = []
        for sent in self.sentences:
            for token_index, token_pair in enumerate(sent):
                token       = token_pair[0]
                pos_tag     = token_pair[1]
                self.features.append(self.get_feature(token, token_index, sent, pos_tag))
                try:
                    ne_label = token_pair[2]
                    self.ne_labels.append(ne_label)
                except:
                    pass
        return

    def train(self, train_datapath):
        """Train named entity recognition model"""
        self.read_data(train_datapath)
        self.form_data()
        print("-> Training phase is started.")
        t0 = time.time()
        self.model.fit(self.vectorizer.fit_transform(self.features), self.ne_labels)
        print("-> Training is completed in %s secs." % (str(round(time.time() - t0, 3))))
        preds = self.model.predict(self.vectorizer.transform(self.features))
        acc_score = accuracy_score(self.ne_labels, preds)
        print("## Evaluation accuracy is %.2f on '%s'" % (acc_score, train_datapath))
        print()
        return

    def evaluate(self, datapath):
        """Evaluate the accuracy of trained named entity recognizer on given development/test corpus data"""
        self.read_data(datapath)
        self.form_data()
        preds       = self.model.predict(self.vectorizer.transform(self.features))
        acc_score   = accuracy_score(self.ne_labels, preds)
        print("## Evaluation accuracy is %.2f on '%s'" % (acc_score, datapath))
        print()
        return acc_score

    def test(self, datapath):
        """Measure various score values of named entity recognizer on given development/test corpus data"""
        self.read_data(datapath)
        self.form_data()
        preds       = self.model.predict(self.vectorizer.transform(self.features))
        precision   = precision_score(self.ne_labels, preds, average='micro')
        recall      = recall_score(self.ne_labels, preds, average='micro')
        f1          = f1_score(self.ne_labels, preds, average='micro')
        accuracy    = accuracy_score(self.ne_labels, preds)
        conf_matrix = confusion_matrix(self.ne_labels, preds)
        return precision, recall, f1, accuracy, conf_matrix

    def tag(self, sentence):
        """Tag single sentence"""
        self.sentences = list([pos_tag(sentence)])
        self.form_data()
        preds       = (self.model.predict(self.vectorizer.transform(self.features)))
        tagged_sent = list(zip(sentence, preds))
        return tagged_sent

    def tag_sents(self, sentences):
        """Tag multiple sentences"""
        tagged_sents = list()
        for sent in sentences:
            tagged_sents.append(self.tag(sent))
        return tagged_sents

    def save(self, save_path):
        """Save named entity recognizer"""
        with gzip.GzipFile(save_path, 'wb') as outfile:
            joblib.dump((self.vectorizer, self.model), outfile, compress=('gzip', 9))
        print("-> Named entity recognizer is saved to '%s'" % save_path)
        return

    def load(self, load_path):
        """Load named entity recognizer"""
        with gzip.GzipFile(load_path, 'rb') as infile:
            self.vectorizer, self.model = joblib.load(infile)
        print("-> Named entity recognizer is loaded from '%s'" % load_path)
        return

def main():

    # SET CORPUS DATA PATHS
    TRAIN_DATAPATH  = '../data/eng.train.txt'
    DEV_DATAPATH    = '../data/eng.testa.txt'

    # INITIALIZE NAMED ENTITY RECOGNIZER
    ner  = NamedEntityRecognizer()

    # TRAIN NAMED ENTITY RECOGNIZER WITH TRAINING CORPUS
    ner.train(TRAIN_DATAPATH)

    # EVALUATE (GET ACCURACY OF) NAMED ENTITY RECOGNIZER ON DEVELOPMENT CORPUS
    ner.evaluate(DEV_DATAPATH)

    # MEASURE SCORES OF NAMED ENTITY RECOGNIZER ON DEVELOPMENT CORPUS AND DISPLAY SCORES
    precision, recall, f1, accuracy, confusion = ner.test(DEV_DATAPATH)
    print('test pre:', precision)
    print('test rec:', recall)
    print('test f1: ', f1)
    print('test acc:', accuracy)
    print('test con:', confusion)

    # SAVE NAMED ENTITY RECOGNIZER
    SAVE_PATH   = '../model/ner_model.gz'
    ner.save(SAVE_PATH)

    # LOAD NAMED ENTITY RECOGNIZER
    LOAD_PATH   = '../model/ner_model.gz'
    ner      = NamedEntityRecognizer()
    ner.load(LOAD_PATH)

    # TAG SINGLE SENTENCE
    print(ner.tag(['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']))

    # TAG MULTIPLE SENTENCES
    print(ner.tag_sents([['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'], \
                            ['BayerVB', 'sets', 'C$', '100', 'million', 'six-year', 'bond', '.']]))

    return

if __name__ == '__main__':
    main()