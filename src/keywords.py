from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
from functools import cached_property
from sklearn.feature_extraction.text import CountVectorizer
import torch
import tensorflow_hub
import re
import numpy as np

from .keybert import KeyBERT  # import local copy of keybert


supported_models = [
    "sentence-transformers/LaBSE",
    "bert-base-uncased",
    "roberta-base",
    "sentence-transformers/all-MiniLM-L6-v2",
    "use",
]


class Keywords:
    def __init__(
        self,
        bert_model: AutoModel,
        tokenizer: AutoTokenizer,
        model_string="bert-base-uncased",
    ):
        if model_string == "use":
            embedding_model = tensorflow_hub.load(
                "https://tfhub.dev/google/universal-sentence-encoder/4"
            )
            self.kw_model = KeyBERT(model=embedding_model)

        self.model_string = model_string
        if model_string not in supported_models:
            self.model_string = "bert-base-uncased"

        self.kw_model = KeyBERT(model=self.model_string)
        self.bert_model = bert_model
        self.tokenizer = tokenizer

        count_vectorizer = CountVectorizer()
        self.count_tokenizer = count_vectorizer.build_tokenizer()

    def get_word_embedding(
        self, sentence: str, word: str, index: int = None
    ) -> torch.tensor:
        # remove subsequent spaces
        clean_sentence = re.sub(" +", " ", sentence)
        tokenized_sentence = self.count_tokenizer(clean_sentence)
        if index is None:
            for i, _word in enumerate(tokenized_sentence):
                if _word.lower() == word.lower():
                    index = i
                    break

        # assert index is not None, f"Error: word {word} not found in provided sentence."
        if index is None:
            return False
        tokens = self.tokenizer(clean_sentence, return_tensors="pt")

        token_ids = [np.where(np.array(tokens.word_ids()) == idx) for idx in [index]]

        with torch.no_grad():
            output = self.bert_model(**tokens)

        # Only grab the last hidden state
        hidden_states = output.hidden_states[-1].squeeze()

        # Select the tokens that we're after corresponding to the word provided
        embedding = hidden_states[[tup[0][0] for tup in token_ids]]
        return embedding

    """uses keybert to generate keywords from a sentence. Gets word embeddings from BERT.
    ex: [('continual', 0.6023), ('change', 0.4642), ('life', 0.4436), ('essence', 0.3975)]
    """

    def get_keywords_with_embeddings(
        self, data: str
    ) -> List[Tuple[str, float, torch.Tensor]]:
        keywords = self.kw_model.extract_keywords(data, keyphrase_ngram_range=(1, 1))

        keywords_with_embeddings = []
        for kw in keywords:
            # # only add the word if it's not numeric
            # if not kw[0].isnumeric():
            embedding = self.get_word_embedding(data, kw[0])
            # can't find the word, proceeed without it.
            if embedding is not False:
                keywords_with_embeddings.append((kw[0], kw[1], embedding))

        # sort by descending to have the most important words first
        desc_sorted_words = sorted(keywords_with_embeddings, key=lambda x: x[1])[::-1]
        return desc_sorted_words

    def get_batch_keywords_with_embeddings(
        self, data: List[str]
    ) -> List[Tuple[str, float, torch.Tensor]]:
        keywords = self.kw_model.extract_keywords(data, keyphrase_ngram_range=(1, 1))

        batch_sentences = []

        for sentence in keywords:
            keywords_with_embeddings = []
            for kw in sentence:
                # # only add the word if it's not numeric
                # if not kw[0].isnumeric():
                embedding = self.get_word_embedding(data, kw[0])
                # can't find the word, proceeed without it.
                if embedding is not False:
                    keywords_with_embeddings.append((kw[0], kw[1], embedding))
            # sort by descending to have the most important words first
            desc_sorted_words = sorted(keywords_with_embeddings, key=lambda x: x[1])[
                ::-1
            ]
            batch_sentences.append(desc_sorted_words)

        return batch_sentences

    """uses keybert to generate keywords from a sentence, returns keybert based word embeddings
    ex: [('continual', 0.6023), ('change', 0.4642), ('life', 0.4436), ('essence', 0.3975)]
    """

    def get_keywords_with_kb_embeddings(
        self, data: str
    ) -> List[Tuple[str, float, torch.Tensor]]:

        keywords_with_embeddings = self.kw_model.extract_keywords(
            data,
            keyphrase_ngram_range=(1, 1),
        )

        # # NON NUMERIC ADDITIONS
        # for kw, we in zip(keywords, word_embeddings):
        #     # all the keywords are numbers, just return them and move on.
        #     if len([kw[0].isnumeric() for kw in keywords_with_embeddings]) == len(
        #         keywords_with_embeddings
        #     ):
        #         keywords_with_embeddings.append((kw[0], kw[1], torch.tensor(we)))
        #     else:
        #         # only add the word if it's not numeric
        #         if not kw[0].isnumeric():
        #             keywords_with_embeddings.append((kw[0], kw[1], torch.tensor(we)))

        # sort by descending to have the most important words first
        desc_sorted_words = sorted(keywords_with_embeddings, key=lambda x: x[1])[::-1]
        return desc_sorted_words

    def get_batch_keywords_with_kb_embeddings(
        self, data: str
    ) -> List[Tuple[str, float, torch.Tensor]]:

        keywords_with_embeddings = self.kw_model.extract_keywords(
            data,
            keyphrase_ngram_range=(1, 1),
        )

        batch_sentences = []

        # # NON NUMERIC ADDITIONS
        # for kw, we in zip(keywords, word_embeddings):
        #     # all the keywords are numbers, just return them and move on.
        #     if len([kw[0].isnumeric() for kw in keywords_with_embeddings]) == len(
        #         keywords_with_embeddings
        #     ):
        #         keywords_with_embeddings.append((kw[0], kw[1], torch.tensor(we)))
        #     else:
        #         # only add the word if it's not numeric
        #         if not kw[0].isnumeric():
        #             keywords_with_embeddings.append((kw[0], kw[1], torch.tensor(we)))

        for sentences in keywords_with_embeddings:
            # sort by descending to have the most important words first
            desc_sorted_words = sorted(sentences, key=lambda x: x[1])[::-1]
            batch_sentences.append(desc_sorted_words)

        return batch_sentences

    def get_keywords(
        self, data: str, emb: bool = True
    ) -> List[Tuple[str, float, torch.Tensor]]:
        return self.kw_model.extract_keywords(
            data, keyphrase_ngram_range=(1, 1), stop_words=None
        )

    def get_keyphrases(
        self, data: str, min_ngram=2, max_ngram=3
    ) -> List[Tuple[str, float]]:
        return self.kw_model.extract_keywords(
            data, keyphrase_ngram_range=(min_ngram, max_ngram), stop_words=None
        )
