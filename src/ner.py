import sys

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

from train import NamedEntityRecognizer


def main(argv):
    if len(argv)!=1:
        print("Usage: python3 ner.py input-sentence")
        exit()

    # READ USER INPUT SENTENCE
    sent        = argv[0]

    # TOKENIZE INPUT SENTENCE
    sent        = word_tokenize(sent)

    # LOAD TRAINED NAMED ENTITY RECOGNIZER
    LOAD_PATH   = '../model/ner_model.gz'
    ner      = NamedEntityRecognizer()
    ner.load(LOAD_PATH)

    # DISPLAY TAGGED SENTENCE
    print(ner.tag(sent))

    return

if __name__ == '__main__':
    main(sys.argv[1:])
