import json
import os
import random
import numpy as np
import pandas as pd
from nltk import word_tokenize

from UniTok import UniTok, Column, Vocab, UniDep
from UniTok.tok import BertTok, IdTok, EntTok, SeqTok, NumberTok, BaseTok

class GloveTok(BaseTok):
    def __init__(self, name: str, path: str):
        super().__init__(name)
        self.vocab = Vocab("danish").load(path, as_path=True)

    def t(self, obj: str):
        ids = []
        objs = word_tokenize(str(obj).lower())
        for o in objs:
            if o in self.vocab.obj2index:
                ids.append(self.vocab.obj2index[o])
        return ids or [self.vocab.obj2index[","]]


class Processor:
    def __init__(self, data_dir, store_dir, glove=None, imp_list_path: str = None):
        self.data_dir = data_dir
        self.store_dir = store_dir
        self.glove = glove
        self.imp_list = json.load(open(imp_list_path, "r")) if imp_list_path else None

        os.makedirs(self.store_dir, exist_ok=True)

        self.train_store_dir = os.path.join(self.store_dir, "train")
        self.dev_store_dir = os.path.join(self.store_dir, "dev")

        self.nid = Vocab(name="article_id")
        self.uid = Vocab(name="user_id")

    def read_news_data(self, mode):
        columns_to_include = ["article_id", "title", "subtitle", "last_modified_time", "premium",
                               "body", "published_time", "image_ids", "article_type", "url", "ner_clusters",
                                 "entity_groups", "topics", "category", "subcategory", "category_str", "total_inviews",
                                   "total_pageviews", "total_read_time", "sentiment_score", "sentiment_label"]

        df = pd.read_parquet(
            os.path.join(self.data_dir, mode, "../preprocessed_and_title_enhanced.parquet"),
            columns=["article_id", "title", "subtitle", "body"]
        )
        return df

    def read_user_data(self, mode):
        df = pd.read_parquet(
            os.path.join(self.data_dir, mode, "behaviours.parquet"),
            columns=["user_id", "article_ids_clicked"]
        )
        return df

    def _read_inter_data(self, mode):
        return pd.read_parquet(
            os.path.join(self.data_dir, mode, "behaviours.parquet"),
            columns=["impression_id", "article_id",  "user_id", "article_ids_inview", "article_ids_clicked"]
        )

    def read_inter_data(self, mode) -> pd.DataFrame:
        df = self._read_inter_data(mode)

        data = dict(impression_id=[], user_id=[], article_id=[], article_ids_clicked=[], max_length_article_ids_clicked=0)
        max_length_of_article_ids_clicked= max(len(x) for x in df["article_ids_inview"])
        data["max_length_article_ids_clicked"] = max_length_of_article_ids_clicked

        for line in df.itertuples():
            clicked_articles = line.article_ids_clicked
            full_interaction = line.article_ids_inview 
            # data["max_length_article_ids_clicked"].extend([max_length_of_article_ids_clicked]* len(full_interaction))
            data["impression_id"].extend([line.impression_id] * len(full_interaction))
            data["user_id"].extend([line.user_id] * len(full_interaction))
            for predict in full_interaction:
                data["article_id"].append(predict)
                if predict in clicked_articles:
                    data["article_ids_clicked"].append(1)
                else:
                    data["article_ids_clicked"].append(0)

        return pd.DataFrame(data)

    def get_news_tok(self, max_title_len=0, max_abs_len=0):
        if self.glove:
            txt_tok = GloveTok(name="english", path=self.glove)
        else:
            txt_tok = BertTok(name="english", vocab_dir="bert-base-uncased")

        return (
            UniTok()
            .add_col(Column(tok=IdTok(vocab=self.nid)))
            .add_col(Column(name="title", tok=txt_tok, max_length=max_title_len))
            .add_col(Column(name="subtitle", tok=txt_tok, max_length=max_abs_len))
            .add_col(Column(name="body", tok=txt_tok, max_length=max_abs_len))
        )
    
    def get_user_tok(self, max_history: int = 0):
        user_ut = UniTok()
        user_ut.add_col(Column(tok=IdTok(vocab=self.uid))).add_col(
            Column(
                name="article_ids_clicked",
                tok=SeqTok(name="article_ids_clicked") 
            )
        )
        return user_ut


    def get_inter_tok(self):
        return (
            UniTok()
            .add_index_col(name="index")
            .add_col(Column(name="impression_id",tok=EntTok))
            .add_col(Column(tok=EntTok(vocab=self.uid)))
            .add_col(Column(tok=EntTok(vocab=self.nid)))
            .add_col(Column(tok=NumberTok(name="article_ids_clicked", vocab_size=2)))
            .add_col(Column(tok=NumberTok(name="max_length_article_ids_clicked", vocab_size=101))) 
        )


    def reassign_inter_df_v2(self):
        inter_train_df = self.read_inter_data("train")
        inter_df = self.read_inter_data("validation")

        inter_dev_df = []
        inter_groups = inter_df.groupby("impression_id")
        for _, imp_df in inter_groups:
            inter_dev_df.append(imp_df)

        return (
            inter_train_df,
            pd.concat(inter_dev_df, ignore_index=True),
        )
    
    def tokenize(self):
        news_tok = self.get_news_tok(max_title_len=20, max_abs_len=50)
        news_df = self.read_news_data("train")

        news_tok.read_file(news_df).tokenize().store_data(
            os.path.join(self.store_dir, "news")
        )

        user_tok = self.get_user_tok(max_history=30)
        user_df = self.read_user_data("train")
        user_tok.read(user_df).tokenize().store(os.path.join(self.store_dir, "user"))

        inter_dfs = self.reassign_inter_df_v2()
        for inter_df, mode in zip(inter_dfs, ["train", "validation"]):
            inter_tok = self.get_inter_tok()
            inter_tok.read_file(inter_df).tokenize().store_data(
                os.path.join(self.store_dir, mode)
            )
    

if __name__ == "__main__":
    p = Processor(
        data_dir="ebnerd-benchmark/data",
        store_dir="ebnerd-benchmark/data/tokenized_bert",
        glove=False
    )

    p.tokenize()