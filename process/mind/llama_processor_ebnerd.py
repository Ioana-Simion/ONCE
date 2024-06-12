import os
import numpy as np
import pandas as pd
from transformers import LlamaTokenizer
from unitok import UniTok 
from column import Column
from vocab import Vocab
from tok.tok import BaseTok
from tok.id_tok import IdTok
from tok.seq_tok import SeqTok
from tok.number_tok import NumberTok


class LlamaTok(BaseTok):
    return_list = True

    def __init__(self, name, vocab_dir):
        super(LlamaTok, self).__init__(name=name)
        self.tokenizer = LlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path=vocab_dir
        )
        vocab = [
            self.tokenizer.convert_ids_to_tokens(i)
            for i in range(self.tokenizer.vocab_size)
        ]
        self.vocab.extend(vocab)

    def t(self, obj) -> [int, list]:
        if pd.notnull(obj):
            ts = self.tokenizer.tokenize(obj)
            ids = self.tokenizer.convert_tokens_to_ids(ts)
        else:
            ids = []
        return ids


class ClickTok(BaseTok):
    def __init__(self, name: str):
        super().__init__(name)
        self.vocab.append(0)
        self.vocab.append(1)
        self.vocab.deny_edit()

    def t(self, obj):
        return int(obj)


class Processor:
    def __init__(self, data_dir, store_dir):
        self.data_dir = data_dir
        self.store_dir = store_dir
        self.v2 = True

        os.makedirs(self.store_dir, exist_ok=True)

        self.nid = Vocab(name="article_id")
        self.topic_voc = Vocab(name="topics")
        # self.total_inviews_voc = Vocab(name="total_inviews")
        # self.total_pageviews_voc = Vocab(name="total_pageviews")

    def read_news_data(self, mode):
        columns_to_include = ["article_id", "title", "subtitle", "body", "category_str", "article_type", "topics", "total_inviews", "total_pageviews"]
        #TODO: "published_time", "last_modified_time"]
                              
        df = pd.read_parquet(
            os.path.join(self.data_dir, mode, "../articles.parquet"),
            columns=columns_to_include
        )
        return df

    def get_news_tok(self, max_title_len=0, max_subtitle_len=0, max_body_len=0, max_category_len=0, max_article_type_len=0, max_topics_len=0):
        txt_tok = LlamaTok(name="llama", vocab_dir="llama_converted")

        return (
            UniTok()
            .add_col(Column(tok=IdTok(vocab=self.nid)))
            .add_col(Column(name="title", tok=txt_tok, max_length=max_title_len))
            .add_col(Column(name="subtitle", tok=txt_tok, max_length=max_subtitle_len))
            .add_col(Column(name="body", tok=txt_tok, max_length=max_body_len))
            .add_col(Column(name="category_str", tok=txt_tok, max_length=max_category_len))
            .add_col(Column(name="article_type", tok=txt_tok, max_length=max_article_type_len))
            .add_col(Column(name="topics", tok=SeqTok(vocab=self.topic_voc), max_length=max_topics_len))
            .add_col(Column(name="total_inviews", tok=NumberTok(vocab_size=4138599, name="total_inviews")))
            .add_col(Column(name="total_pageviews", tok=NumberTok(vocab_size=1637752, name="total_pageviews")))
        )

    def analyse_news(self):
        tok = self.get_news_tok(max_title_len=0, max_subtitle_len=0, max_body_len=0, max_category_len=0, max_article_type_len=0, max_topics_len=0)
        df = self.combine_news_data()
        tok.read_file(df).analyse()

    def tokenize(self):
        news_tok = self.get_news_tok(max_title_len=20, max_subtitle_len=20, max_body_len=400, max_category_len=20, max_article_type_len=20, max_topics_len=100)
        news_df =  self.read_news_data("train")

        news_tok.read_file(news_df).tokenize().store_data(
            os.path.join(self.store_dir)
        )

if __name__ == "__main__":
    p = Processor(
        data_dir="ebnerd-benchmark/data",
        store_dir="ebnerd-benchmark/data/tokenized_llama",
    )

    p.tokenize()