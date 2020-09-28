import numpy as np
import pathlib
import pickle
from ..cover_tokens import cover_tokens_new as cover_tokens
from ..cover_tokens import word_tokenizer as simple_tokenizer
from ..cover_tokens import regroup_attributions
from ..cover_tokens import strip_accents_and_special_characters

# TODO - Raghu's suggestions
# - consider unittest for tests rather than pytest.
# - to simplify tests, remove dependence on tokenizers and provide token-lists
#   directly.

PACKAGE_PATH = pathlib.Path(__file__).parents[1]
BERT_TOKENIZER_PATH = PACKAGE_PATH / 'bert_tokenizer.pkl'
RNN_TOKENIZER_PATH = PACKAGE_PATH / 'imdb_rnn_tokenizer.pkl'

with open(BERT_TOKENIZER_PATH, 'rb') as handle:
    bert_tokenizer = pickle.load(handle)

# Wrap the imdb_rnn_tokenizer so it can 'tokenize' like BERT
with open(RNN_TOKENIZER_PATH, 'rb') as handle:
    imdb_rnn_tokenizer = pickle.load(handle)
    imdb_rnn_tokenizer.tokenize = (
        lambda x: list(imdb_rnn_tokenizer.decode([y]) for y in
                       imdb_rnn_tokenizer.encode(x)))


# Tuples of name, word and wordpiece tokenizers, and a text-to-text
# pre-processing function to test together (in case e.g. you need to
# strip unicode specials).
tokenizers = [('BERT',
               bert_tokenizer.basic_tokenizer.tokenize,
               bert_tokenizer.tokenize,
               lambda x: x),
              ('imdb_rnn',
               simple_tokenizer,
               imdb_rnn_tokenizer.tokenize,
               strip_accents_and_special_characters
               )]


def check_token_covering(token_covering,
                         coarse_grained_tokens,
                         fine_grained_tokens):
    covered_coarse_grained_tokens = []
    covered_fine_grained_tokens = []
    for t, tokens in token_covering:
        covered_coarse_grained_tokens.append(t)
        covered_fine_grained_tokens += tokens

    # Test that all coarse-grained tokens are present in the covering.
    assert(np.array_equal(covered_coarse_grained_tokens,
                          coarse_grained_tokens))

    # Test that the covering covers all fine-grained tokens.
    assert(np.array_equal(covered_fine_grained_tokens,
                          fine_grained_tokens))


def test_space_separated_tokens_with_wordpiece_covering():

    for tok_name, _, wordpiece_tokenize, _ in tokenizers:

        print(f'Testing tokenizer: {tok_name}')

        sentence = 'this is a test sentence with a crazycomplexword and ' \
                   "'punctuation,'!!!"

        space_tokens = sentence.split(' ')
        wordpiece_tokens = wordpiece_tokenize(sentence)

        token_covering = cover_tokens(
            coarse_grained_tokens=space_tokens,
            fine_grained_tokens=wordpiece_tokens)

        assert(token_covering is not None)
        check_token_covering(token_covering, space_tokens, wordpiece_tokens)


def test_word_with_wordpiece_covering():

    sentence = 'this is a test sentence with a crazycomplexword and ' \
               "'punctuation,'!!!"

    for tok_name, word_tokenize, wordpiece_tokenize, _ in tokenizers:

        word_tokens = word_tokenize(sentence)
        wordpiece_tokens = wordpiece_tokenize(sentence)
        token_covering = cover_tokens(
            coarse_grained_tokens=word_tokens,
            fine_grained_tokens=wordpiece_tokens)

        assert(token_covering is not None)
        check_token_covering(token_covering, word_tokens, wordpiece_tokens)


# For BERT only– imdb_rnn produces a technically correct (but admittedly odd)
# cover.
def test_crazy_tokens_covering():
    tokenizer = bert_tokenizer

    sentence = 'this is a test sentence'
    crazy_tokens = ['thi', 's a t', 'est sent']
    wordpiece_tokens = tokenizer.tokenize(sentence)
    token_covering = cover_tokens(
        coarse_grained_tokens=crazy_tokens,
        fine_grained_tokens=wordpiece_tokens)

    assert(token_covering is None)


# Tokenizers don't handle unicode special characters the same way (accents,
# emojis, etc.)  In some cases we'll have to wrap strings in a preprocessor
# to strip specials before tokenizing (e.g. imdb_rnn).
def test_special_chars():
    sentence = "It's naïve to think we've visited every café in town."

    for tok_name, word_tokenize, wordpiece_tokenize, pre_process in tokenizers:
        print(tok_name)

        word_tokens = word_tokenize(pre_process(sentence))
        wordpiece_tokens = wordpiece_tokenize(pre_process(sentence))
        token_covering = cover_tokens(
            coarse_grained_tokens=word_tokens,
            fine_grained_tokens=wordpiece_tokens)

        assert(token_covering is not None)
        check_token_covering(token_covering, word_tokens, wordpiece_tokens)


def incomplete_tokenization():
    for tok_name, word_tokenize, wordpiece_tokenize in tokenizers:
        print(f'Testing {tok_name}')
        sentence = 'this is a test sentence with a crazycomplexword and ' \
                   "'punctuation,'!!!"
        word_tokens = word_tokenize(sentence)[:-3]  # dropping some word tokens
        wordpiece_tokens = wordpiece_tokenize(sentence)
        token_covering = cover_tokens(
            coarse_grained_tokens=word_tokens,
            fine_grained_tokens=wordpiece_tokens)

        assert(token_covering is None)


def test_regroup_attributions():
    sentence = 'this is a test sentence with a crazycomplexword and ' \
               "'punctuation,'!!!"

    word_tokens = bert_tokenizer.basic_tokenizer.tokenize(sentence)
    wordpiece_tokens = bert_tokenizer.tokenize(sentence)

    wordpiece_attributions = list(range(1, len(wordpiece_tokens)+1))

    coverings = cover_tokens(word_tokens, wordpiece_tokens)

    # Did we just generate an attribution value for each wordpiece?
    assert len(wordpiece_attributions) == len(wordpiece_tokens)

    word_attributions = regroup_attributions(coverings, wordpiece_attributions)

    # Is all the attribution value retained after regrouping?
    assert sum(word_attributions) == sum(wordpiece_attributions)

    # Do we have one attribution per word token?
    assert len(word_tokens) == len(word_attributions)
