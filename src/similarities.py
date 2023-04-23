from transformers import AutoTokenizer, AutoModel
from functools import cached_property
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import torch
import re


class Embedding:
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

        count_vectorizer = CountVectorizer()
        self.count_tokenizer = count_vectorizer.build_tokenizer()

    def get_word_embedding(self, sentence: str, word: str, index: int = None) -> torch.tensor:
        # remove subsequent spaces
        clean_sentence = re.sub(' +', ' ', sentence)
        tokenized_sentence = self.count_tokenizer(clean_sentence)
        if index is None:
            for i, _word in enumerate(tokenized_sentence):
                if _word.lower() == word.lower():
                    index = i
                    break

        assert index is not None, "Error: word not found in provided sentence."
        tokens = self.tokenizer(clean_sentence, return_tensors='pt')

        token_ids = [np.where(np.array(tokens.word_ids()) == idx)
                     for idx in [index]]

        with torch.no_grad():
            output = self.model(**tokens)

        # Only grab the last hidden state
        hidden_states = output.hidden_states[-1].squeeze()

        # Select the tokens that we're after corresponding to the word provided
        embedding = hidden_states[[tup[0][0] for tup in token_ids]]
        return embedding

    def get_similarity(self, emb1: torch.tensor, emb2: torch.tensor) -> torch.tensor:
        return torch.cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))

    # utility function to create comparisons between two rows of keywords
    # this function includes the keywords with the embeddings
    def compare_keyword_tuples(
        self,
        first_keywords: List[Tuple[str, float, torch.Tensor]],
        second_keywords: List[Tuple[str, float, torch.Tensor]],
        suppress_errors=True
    ):
        word_comparisons = []
        for word_tuple in first_keywords:
            word = word_tuple[0]
            for second_word_tuple in second_keywords:
                second_word = second_word_tuple[0]

                try:
                    word_one_emb = word_tuple[2]
                    word_two_emb = second_word_tuple[2]

                    word_comparisons.append(
                        (
                            word,
                            second_word,
                            self.get_similarity(word_one_emb, word_two_emb),
                        )
                    )
                except AssertionError as e:
                    if not suppress_errors:
                        print(e, word, second_word)

        return word_comparisons

    def compare_keywords(
        self,
        first_sentence: str,
        second_sentence: str,
        first_keywords: List[str],
        second_keywords: List[str],
        suppress_errors=True
    ):
        word_comparisons = []
        for word in first_keywords:
            for second_word in second_keywords:

                try:
                    word_one_emb = self.get_word_embedding(
                        first_sentence, word)
                    word_two_emb = self.get_word_embedding(
                        second_sentence, second_word
                    )

                    word_comparisons.append(
                        (
                            word,
                            second_word,
                            self.get_similarity(word_one_emb, word_two_emb),
                        )
                    )
                except AssertionError as e:
                    if not suppress_errors:
                        print(e, second_word, second_sentence)

        return word_comparisons


class Similarities:
    """create instance of similarities
    model_string: type of model from HuggingFace (e.g., 'bert-base-uncased')
    """

    def __init__(self, model_string: str = 'bert-base-uncased'):
        self.model_string = model_string

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_string)

    @cached_property
    def model(self):
        return AutoModel.from_pretrained(self.model_string, output_hidden_states=True).eval()
