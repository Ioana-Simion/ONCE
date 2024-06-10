import json
import os
import random

import numpy as np
import pandas as pd
from nltk import word_tokenize
from UniTok import Column, UniTok, Vocab
from UniTok.tok import BaseTok, BertTok, EntTok, IdTok, NumberTok, SplitTok


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
            columns=columns_to_include
        )
        return df

    def read_user_data(self, mode):
        return pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, mode, "behaviors.tsv"),
            sep="\t",
            names=["imp", "uid", "time", "history", "predict"],
            usecols=["uid", "history"],
        )

    def _read_inter_data(self, mode):
        return pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, mode, "behaviors.tsv"),
            sep="\t",
            names=["imp", "uid", "time", "history", "predict"],
            usecols=["imp", "uid", "predict"],
        )

    def read_neg_data(self, mode):
        df = self._read_inter_data(mode)
        data = dict(uid=[], neg=[])
        for line in df.itertuples():
            if line.uid in data["uid"]:
                continue

            predicts = line.predict.split(" ")
            negs = []
            for predict in predicts:
                nid, click = predict.split("-")
                if not int(click):
                    negs.append(nid)

            data["uid"].append(line.uid)
            data["neg"].append(" ".join(negs))
        return pd.DataFrame(data)

    def read_inter_data(self, mode) -> pd.DataFrame:
        df = self._read_inter_data(mode)
        data = dict(imp=[], uid=[], nid=[], click=[])
        for line in df.itertuples():
            predicts = line.predict.split(" ")
            data["imp"].extend([line.imp] * len(predicts))
            data["uid"].extend([line.uid] * len(predicts))
            for predict in predicts:
                nid, click = predict.split("-")
                data["nid"].append(nid)
                data["click"].append(int(click))
        return pd.DataFrame(data)

    def get_news_tok(self, max_title_len=0, max_abs_len=0):
        if self.glove:
            txt_tok = GloveTok(name="danish", path=self.glove)
        else:
            txt_tok = BertTok(name="danish", vocab_dir="bert-base-uncased")

        return (
            UniTok()
            .add_col(Column(name=self.nid, tok=IdTok(vocab=self.nid)))
            .add_col(Column(name="title", tok=txt_tok, max_length=max_title_len))
            .add_col(Column(name="subtitle", tok=txt_tok, max_length=max_abs_len))
            #.add_col(Column(name="last_modified_time", tok=NumberTok()))  # Assuming timestamp as number for simplicity
            #.add_col(Column(name="premium", tok=IdTok(vocab=Vocab("bool_vocab"))))  # Map True/False to IDs
            .add_col(Column(name="body", tok=txt_tok, max_length=max_abs_len))
            #.add_col(Column(name="published_time", tok=NumberTok()))  # Assuming timestamp as number
            #.add_col(Column(name="image_ids", tok=SplitTok(sep=" ", vocab=Vocab("image_id_vocab"))))
            #.add_col(Column(name="article_type", tok=EntTok()))  # Assuming article types as entities
            #.add_col(Column(name="url", tok=txt_tok))
            #.add_col(Column(name="ner_clusters", tok=SplitTok(sep=" ", vocab=Vocab("ner_vocab"))))
            #.add_col(Column(name="entity_groups", tok=SplitSubcatTok(sep=" ", vocab=Vocab("entity_vocab"))))
            .add_col(Column(name="topics", tok=SplitTok(sep=" ", vocab=Vocab("topic_vocab"))))
            #.add_col(Column(name="category", tok=NumberTok()))
            #.add_col(Column(name="subcategory", tok=SplitTok(sep=" ", vocab=Vocab("subcat_vocab"))))
            #.add_col(Column(name="category_str", tok=EntTok()))
            #.add_col(Column(name="total_inviews", tok=NumberTok()))
            #.add_col(Column(name="total_pageviews", tok=NumberTok()))
            #.add_col(Column(name="total_read_time", tok=NumberTok()))
            #.add_col(Column(name="sentiment_score", tok=NumberTok()))
            #.add_col(Column(name="sentiment_label", tok=EntTok()))
        )


    def get_user_tok(self, max_history: int = 0):
        user_ut = UniTok()
        user_ut.add_col(Column(tok=IdTok(vocab=self.uid))).add_col(
            Column(
                name="history",
                tok=SplitTok(sep=" ", vocab=self.nid),
                max_length=max_history,
                slice_post=True,
            )
        )
        return user_ut

    def get_neg_tok(self, max_neg: int = 0):
        neg_ut = UniTok()
        neg_ut.add_col(
            Column(
                tok=IdTok(vocab=self.uid),
            )
        ).add_col(
            Column(
                name="neg",
                tok=SplitTok(sep=" ", vocab=self.nid),
                max_length=max_neg,
                slice_post=True,
            )
        )
        return neg_ut

    def get_inter_tok(self):
        return (
            UniTok()
            .add_index_col(name="index")
            .add_col(
                Column(
                    name="imp",
                    tok=EntTok,
                )
            )
            .add_col(Column(tok=EntTok(vocab=self.uid)))
            .add_col(Column(tok=EntTok(vocab=self.nid)))
            .add_col(Column(tok=NumberTok(name="click", vocab_size=2)))
        )

    def combine_news_data(self):
        df = self.read_news_data("train")
        return df

    def combine_user_df(self):
        user_train_df = self.read_user_data("train")
        user_dev_df = self.read_user_data("dev")

        user_df = pd.concat([user_train_df, user_dev_df])
        user_df = user_df.drop_duplicates(["uid"])
        return user_df

    def combine_neg_df(self):
        neg_train_df = self.read_neg_data("train")
        neg_dev_df = self.read_neg_data("dev")

        neg_df = pd.concat([neg_train_df, neg_dev_df])
        neg_df = neg_df.drop_duplicates(["uid"])
        return neg_df

    def combine_inter_df(self):
        inter_train_df = self.read_inter_data("train")
        inter_dev_df = self.read_inter_data("dev")
        inter_dev_df.imp += max(inter_train_df.imp)

        inter_df = pd.concat([inter_train_df, inter_dev_df])
        return inter_df

    def splitter(self, l: list, portions: list):
        if self.imp_list:
            l = self.imp_list
        else:
            random.shuffle(l)
        json.dump(l, open(os.path.join(self.store_dir, "imp_list.json"), "w"))

        portions = np.array(portions)
        portions = portions * 1.0 / portions.sum() * len(l)
        portions = list(map(int, portions))
        portions[-1] = len(l) - sum(portions[:-1])

        pos = 0
        parts = []
        for i in portions:
            parts.append(l[pos : pos + i])
            pos += i
        return parts

    def reassign_inter_df_v2(self):
        inter_train_df = self.read_inter_data("train")
        inter_df = self.read_inter_data("dev")

        imp_list = inter_df.imp.drop_duplicates().to_list()

        dev_imps, test_imps = self.splitter(imp_list, [5, 5])
        inter_dev_df, inter_test_df = [], []

        inter_groups = inter_df.groupby("imp")
        for imp, imp_df in inter_groups:
            if imp in dev_imps:
                inter_dev_df.append(imp_df)
            else:
                inter_test_df.append(imp_df)
        return (
            inter_train_df,
            pd.concat(inter_dev_df, ignore_index=True),
            pd.concat(inter_test_df, ignore_index=True),
        )

    def analyse_news(self):
        tok = self.get_news_tok(max_title_len=0, max_abs_len=0)
        df = self.combine_news_data()
        tok.read(df).analyse()

    def analyse_user(self):
        tok = self.get_user_tok(max_history=0)
        df = self.combine_user_df()
        tok.read(df).analyse()

    def analyse_inter(self):
        tok = self.get_inter_tok()
        df = self.combine_inter_df()
        tok.read_file(df).analyse()

    def tokenize(self):
        news_tok = self.get_news_tok(max_title_len=20, max_abs_len=50)
        news_df = self.combine_news_data()
        news_tok.read_file(news_df).tokenize().store_data(
            os.path.join(self.store_dir, "news")
        )

        user_tok = self.get_user_tok(max_history=30)
        user_df = self.combine_user_df()
        user_tok.read(user_df).tokenize().store(os.path.join(self.store_dir, "user"))

        inter_dfs = self.reassign_inter_df_v2()
        for inter_df, mode in zip(inter_dfs, ["train", "dev", "test"]):
            inter_tok = self.get_inter_tok()
            inter_tok.read_file(inter_df).tokenize().store_data(
                os.path.join(self.store_dir, mode)
            )

    def tokenize_original_dev(self):
        news_tok = self.get_news_tok(max_title_len=20, max_abs_len=50)
        news_df = self.combine_news_data()
        news_tok.read_file(news_df).tokenize()

        user_tok = self.get_user_tok(max_history=30)
        user_df = self.combine_user_df()
        user_tok.read(user_df).tokenize()

        inter_df = self.read_inter_data("dev")
        inter_tok = self.get_inter_tok()
        inter_tok.read_file(inter_df).tokenize().store_data(
            os.path.join(self.store_dir, "dev-original")
        )

    def tokenize_neg(self):
        print("tokenize neg")
        self.uid.load(os.path.join(self.store_dir, "user"))
        self.nid.load(os.path.join(self.store_dir, "news"))

        print("combine neg df")
        neg_df = self.combine_neg_df()
        print("get neg tok")
        neg_tok = self.get_neg_tok()
        neg_tok.read(neg_df).tokenize().store(os.path.join(self.store_dir, "neg"))


if __name__ == "__main__":
    # p = Processor(
    #     data_dir='/data1/qijiong/Data/MIND/',
    #     store_dir='../../data/MIND-small-v3',
    # )
    #
    # p.tokenize()
    # p.tokenize_neg()

    p = Processor(
        data_dir="ebnerd-benchmark\data\ebnerd_small",
        store_dir="../../data/MIND-small-v2",
    )

    p.tokenize_original_dev()
