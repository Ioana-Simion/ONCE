import json
import os
import random

import numpy as np
import pandas as pd
from nltk import word_tokenize

# New imports
from column import Column
from unitok import UniTok
from vocab import Vocab
from tok import BaseTok, BertTok, EntTok, IdTok, NumberTok, SplitTok, SeqTok


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
            os.path.join(self.data_dir, mode, "../articles.parquet"),
            columns=["article_id", "title", "subtitle", "body"]
        )
        return df

    def read_user_data(self, mode):

        column_names = ["article_id", "impression_time", "read_time", "scroll_percentage",
                         "device_type", "article_ids_inview", "article_ids_clicked", "user_id",
                           "is_sso_user", "gender", "postcode", "age", "is_subscriber", "session_id",
                             "next_read_time", "next_scroll_percentage", "__fragment_index", "__batch_index",
                               "__last_in_fragment", "__filename"]


        df = pd.read_parquet(
            os.path.join(self.data_dir, mode, "behaviors.parquet"),
            columns=["user_id", "article_ids_clicked"]
        )
        # Previously: columns=["uid", "history"]. History is defined as: The news click history (ID list of clicked news) 
        # of this user before this impression. The clicked news articles are ordered by time.
        # Therefore, I (Jort) selected the columns "User ID" and "Clicked Article IDs" to be used as the user data.
        return df

    def _read_inter_data(self, mode):

    
        column_names = ["article_id", "impression_time", "read_time", "scroll_percentage",
                         "device_type", "article_ids_inview", "article_ids_clicked", "user_id",
                           "is_sso_user", "gender", "postcode", "age", "is_subscriber", "session_id",
                             "next_read_time", "next_scroll_percentage", "__fragment_index", "__batch_index",
                               "__last_in_fragment", "__filename"]
        
        return pd.read_parquet(
            os.path.join(self.data_dir, mode, "behaviors.parquet"),
            columns=["impression_id", "article_id",  "user_id", "article_ids_inview", "article_ids_clicked"]
        )

    def read_inter_data(self, mode) -> pd.DataFrame:
        df = self._read_inter_data(mode)
        data = dict(impression_id=[], user_id=[], article_id=[], article_ids_clicked=[])
        for line in df.itertuples():
            clicked_articles = line.article_ids_clicked
            full_interaction =line.article_ids_inview 

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
                tok=SeqTok(name="article_ids_clicked")  # Using SeqTok to tokenize the list of clicked article IDs
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
    # p = Processor(
    #     data_dir='/data1/qijiong/Data/MIND/',
    #     store_dir='../../data/MIND-small-v3',
    # )
    #
    # p.tokenize()
    # p.tokenize_neg()

    p = Processor(
        data_dir="ebnerd-benchmark/data",
        store_dir="/ebnerd-benchmark/data",
        glove=False
    )

    p.tokenize()




    
#.add_col(Column(name="last_modified_time", tok=NumberTok()))  # Assuming timestamp as number for simplicity
#.add_col(Column(name="premium", tok=IdTok(vocab=Vocab("bool_vocab"))))  # Map True/False to IDs
#.add_col(Column(name="published_time", tok=NumberTok()))  # Assuming timestamp as number
#.add_col(Column(name="image_ids", tok=SplitTok(sep=" ", vocab=Vocab("image_id_vocab"))))
#.add_col(Column(name="article_type", tok=EntTok()))  # Assuming article types as entities
#.add_col(Column(name="url", tok=txt_tok))
#.add_col(Column(name="ner_clusters", tok=SplitTok(sep=" ", vocab=Vocab("ner_vocab"))))
#.add_col(Column(name="entity_groups", tok=SplitSubcatTok(sep=" ", vocab=Vocab("entity_vocab"))))
#.add_col(Column(name="topics", tok=SplitTok(sep=" ", vocab=Vocab("topic_vocab"))))
#.add_col(Column(name="category", tok=NumberTok()))
#.add_col(Column(name="subcategory", tok=SplitTok(sep=" ", vocab=Vocab("subcat_vocab"))))
#.add_col(Column(name="category_str", tok=EntTok()))
#.add_col(Column(name="total_inviews", tok=NumberTok()))
#.add_col(Column(name="total_pageviews", tok=NumberTok()))
#.add_col(Column(name="total_read_time", tok=NumberTok()))
#.add_col(Column(name="sentiment_score", tok=NumberTok()))
#.add_col(Column(name="sentiment_label", tok=EntTok()))
