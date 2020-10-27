
import numpy as np
from typing import Callable
from unidecode import unidecode


def strip_accents_and_special_characters(s):
    return unidecode(s)


def one_split(in_strings: list,
              split_string: str,
              strip_whitespace: bool) -> list:
    """Break each string in a list of strings into smaller parts.

    Split after each occurrence of split_string.

    :param in_strings: List of strings to be broken into substrings.
    :param split_string: A separator string after which to divide.
    :param strip_whitespace:(bool) leading/trailing whitespace from
        tokens.
    :return: A list of (probably) smaller strings.
    """
    out = []

    include_delim = split_string.strip() if strip_whitespace else split_string

    for sub_str in in_strings:
        splits = sub_str.split(split_string)
        for piece in splits:
            if strip_whitespace:
                piece = piece.strip()
            if piece:
                out.append(piece)
            if include_delim:
                out.append(split_string)
        if include_delim:
            out.pop()
    return out


def multi_split(in_strings: str,
                split_strings: tuple,
                strip_whitespace: bool) -> list:
    """
    Split strings in a list at any of multiple split-strings.

    :param in_strings:  List of strings to be broken into substrings.
    :param split_strings: List of string separators after which to
        divide.
    :param strip_whitespace: Remove leading/trailing whitespace from
        tokens?
    :return: A list of (probably) smaller strings.
    """
    out = [in_strings]
    for split_string in split_strings:
        out = one_split(out, split_string, strip_whitespace)
    return out


def word_tokenizer(raw_string: str,
                   delimiters: tuple =
                   (' ', '.', ',', '>', '!', ';', ':', '--'),
                   strip_whitespace: bool = False) -> list:
    """
    Simple tokenizer that splits on spaces and assorted punctuation.
    Also retains separators.

    :param raw_string: string to tokenize
    :param delimiters: [(' ', '.', ',', '>', '!', ';', ':', '--')]
        list of string splitting delimiters.
    :param strip_whitespace: [False] Remove leading/trailing whitespace
        from tokens?
    :return: List of substrings
    """
    return multi_split(raw_string, delimiters, strip_whitespace)


def cover_tokens(coarse_grained_tokens: list,
                 fine_grained_tokens: list,
                 num_fine_tokens_to_be_matched=None) -> list:
    """
    Given two tokenizations of a sentence -- one coarse-grained (e.g.
    word-level tokenization), and one fine-grained (e.g., wordpiece-
    level tokenization), this method returns a covering of the
    coarse-grained tokens with fine-grained tokens.

    Specifically, the returned covering comes with the guarantee that
    the concatenation of the lists of fine-grained tokens assigned to
    each coarse-grained token would recover the original list of
    fine-grained tokens.

    Additionally, the fine-grained tokens may include additional
    characters (which some tokenizers create), but MUST CONTAIN all the
    characters from the concatenated coarse-grained tokens in the same
    order (until num_fine_tokens_to_be_matched fine-tokens have been
    processed, if specified).

    Further, a fine-grained token is guaranteed to belong to one and
    only one coarse-grained token and is associated with the first
    coarse-grained token to which it contributes.  It doesn't need to
    end in the same coarse-grained token... this helps to accommodate
    tokenizers that may split whitespace and punctuation differently.

    The method returns None if a suitable covering cannot be defined.

    Example:

      sentence = 'coarse tokens fine.'

      coarse = word_tokenizer(sentence)
      # ['coarse', ' ', 'tokens', ' ', 'fine', '.']

      fine = imdb_rnn_tokenizer(sentence)
      # ['coa', 'rse', ' ', 'to', 'ken', 's ', 'fine', '.']

      cover_tokens_new(coarse, fine)

      # [('coarse', ['coa', 'rse']),
      #  (' ', [' ']),
      #  ('tokens', ['to', 'ken', 's ']),
      #  (' ', []),
      #  ('fine', ['fine']),
      #  ('.', ['.'])]

    Notice that 's ' in the fine tokenization straddles two coarse-
    tokens 'tokens' and ' ', it is associated with the first, but the
    still satisfies the requirement that the second is character-for-
    character matched.

    :param coarse_grained_tokens: List with tokens from a
    coarse-grained tokenization (e.g., word-level tokenization) of the
    input sentence

    :param fine_grained_tokens: List with tokens from a
    fine-grained tokenization (e.g., wordpiece-level or character-level
    tokenization) of the  input sentence.

    :param num_fine_tokens_to_be_matched: [Default None] If None,
    require all characters in coarse-grained tokens to be matched.
    If this is passed an integer, only require this many fine-grained
    tokens to match before declaring the covering valid. Helpful when
    model takes a specific number of input tokens.

    :returns token_covering: List of tuples where the i^th tuple
    consists of the i^th coarse-grained token followed by a list of
    fine-grained tokens it maps to; None if covering isn't possible.
    """
    coverings = []
    num_fine_tokens_processed = 0

    fine_token_iter = iter(fine_grained_tokens)
    coarse_token_iter = iter(coarse_grained_tokens)

    # These init values will kick-off the draw loops for coarse and fine chars
    coarse_char_iter = iter('')
    fine_char_iter = iter('')
    coarse_char = None
    fine_char = None

    # This while loop compares coarse and fine tokens, one character at a time
    # If they match, both step; if not, only the fine character steps this
    # allows the algorithm to skip extra characters that the fine tokenizer
    # might have added.
    #
    # Each time a new fine-grained token is drawn, it is added to the current
    # active coarse grained token right away.
    #
    # If a coarse-grained token ins consumed, a new one is picked up and a new
    # covering entry is created for it.  Any partially processed fine tokens
    # will continue to match characters in the new coarse token.  However
    # the fine-token will continue to be associated with only the previous
    # covering.  See the docstring example for a straddling case with the
    # 's ' fine-grained token.

    while True:
        if coarse_char == fine_char:
            while True:  # Draw coarse_char until valid
                try:
                    coarse_char = next(coarse_char_iter)
                    break
                except StopIteration:  # Need a new token
                    try:
                        coarse_token = next(coarse_token_iter)
                        coverings.append([coarse_token, []])
                        coarse_char_iter = iter(coarse_token)
                    except StopIteration:  # End of coarse tokens
                        return coverings

        # Increment fine_char whether or not there was a match.
        while True:  # Draw fine_char until valid
            try:
                fine_char = next(fine_char_iter)
                break
            except StopIteration:  # Need a new token
                try:
                    fine_token = next(fine_token_iter)
                    num_fine_tokens_processed += 1
                    coverings[-1][1].append(fine_token)
                    fine_char_iter = iter(fine_token)
                except StopIteration:  # End of fine tokens
                    if (num_fine_tokens_to_be_matched and
                            num_fine_tokens_processed >=
                            num_fine_tokens_to_be_matched):
                        return coverings
                    else:
                        return None


def regroup_attributions(coverings: list, fine_attributions: list) -> list:
    """
    Produces a list of len(coverings) summed attributions according to
    the groupings of the tuples in coverings.

     Example:

       covering =[(“simple”, [“simple”]),
                  (“example”, [“exam#”, “#ple”])]

       fine_attributions = [0.1, 0.3. 0.4]

       regroup_attributions(covering, fine_attributions)

       # [ 0.1,  0.7 ] <- one fore each coarse token

    :param coverings: List of tuples grouping tokens together
    :param fine_attributions: List of attribution values, one for
        each entry in the concatenated covering list.
    :return: A list of combined coverings, one for each tuple in
    covering.
    """
    coarse_attributions = []
    offset = 0

    for coarse_token, fine_tokens_covered in coverings:
        num_tokens = len(fine_tokens_covered)
        coarse_attributions.append(sum(fine_attributions[
                                       offset:offset + num_tokens])
                                   if num_tokens else 0.)
        offset += num_tokens
    return coarse_attributions
