from typing import Dict, List
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.token_indexers.elmo_indexer import _make_bos_eos, ELMoTokenCharactersIndexer


class UnicodeELMoCharacterMapper(object):
    max_word_length = 50

    oov_token = '<oov>'
    pad_token = '<pad>'
    bos_token = '<S>'
    eos_token = '</S>'
    bow_token = '<W>'
    eow_token = '</W>'

    def __init__(self, char_vocab_file: str,
                 tokens_to_add: Dict[str, int] = None) -> None:
        self.mapping = {}
        with open(char_vocab_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                fields = line.strip().split('\t')
                assert len(fields) == 2
                token, i = fields
                self.mapping[token] = int(i)

        self.beginning_of_sentence_character = self.mapping.get(self.bos_token)
        self.end_of_sentence_character = self.mapping.get(self.eos_token)
        self.beginning_of_word_character = self.mapping.get(self.bow_token)
        self.end_of_word_character = self.mapping.get(self.eow_token)
        self.padding_character = self.mapping.get(self.pad_token)
        self.oov_character = self.mapping.get(self.oov_token)

        self.beginning_of_sentence_characters = _make_bos_eos(
            self.beginning_of_sentence_character,
            self.padding_character,
            self.beginning_of_word_character,
            self.end_of_word_character,
            self.max_word_length
        )

        self.end_of_sentence_characters = _make_bos_eos(
            self.end_of_sentence_character,
            self.padding_character,
            self.beginning_of_word_character,
            self.end_of_word_character,
            self.max_word_length
        )

        self.tokens_to_add = tokens_to_add or {}

    def convert_word_to_char_ids(self, word: str) -> List[int]:
        if word in self.tokens_to_add:
            char_ids = [self.padding_character] * self.max_word_length
            char_ids[0] = self.beginning_of_word_character
            char_ids[1] = self.tokens_to_add[word]
            char_ids[2] = self.end_of_word_character
        elif word == self.bos_token:
            char_ids = self.beginning_of_sentence_characters
        elif word == self.eos_token:
            char_ids = self.end_of_sentence_characters
        else:
            char_ids = [self.padding_character] * self.max_word_length
            char_ids[0] = self.beginning_of_word_character
            for k, chr_id in enumerate(word, start=1):
                char_ids[k] = self.mapping.get(chr_id, self.oov_character)
            char_ids[len(word) + 1] = self.end_of_word_character

        return char_ids
        # I am not sure if we need this masking.
        # +1 one for masking
        #return [(c + 1) for c in char_ids]


@TokenIndexer.register("elmo_characters_v2")
class UnicodeELMoTokenCharactersIndexer(ELMoTokenCharactersIndexer):
    """
    Convert a token to an array of character ids to compute ELMo representations.

    Parameters
    ----------
    namespace : ``str``, optional (default=``elmo_characters``)
    tokens_to_add : ``Dict[str, int]``, optional (default=``None``)
        If not None, then provides a mapping of special tokens to character
        ids. When using pre-trained models, then the character id must be
        less then 261, and we recommend using un-used ids (e.g. 1-32).
    """
    # pylint: disable=no-self-use
    def __init__(self, char_vocab_file: str,
                 namespace: str = 'elmo_characters_v2',
                 tokens_to_add: Dict[str, int] = None) -> None:
        super(UnicodeELMoTokenCharactersIndexer, self).__init__(namespace, tokens_to_add)
        self._mapper = UnicodeELMoCharacterMapper(char_vocab_file, tokens_to_add)
