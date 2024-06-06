import json
import os

import pandas as pd
from tqdm import tqdm
from UniTok import UniDep


class GoodreadsPrompter:
    def __init__(self, data_path, desc_path=None, v2=False):
        self.data_path = data_path
        self.desc_path = desc_path
        self.v2 = v2

        self.book_df = pd.read_csv(
            filepath_or_buffer=os.path.join(data_path),
            sep="\t",
            header=0,
        )

        self.use_desc = desc_path is not None
        if desc_path:
            self.desc_df = pd.read_csv(
                filepath_or_buffer=os.path.join(desc_path),
                sep=",",
                header=0,
            )
            self.book_df = pd.merge(self.book_df, self.desc_df, on="bid")

        self._book_list = None
        self._book_dict = None

    def stringify(self):
        if self._book_list is not None:
            return self._book_list
        self._book_list = []
        for news in tqdm(self.book_df.iterrows()):
            if self.use_desc:
                if self.v2:
                    string = (
                        f'title: {news[1]["title"]}\ndescription: {news[1]["desc"]}\n'
                    )
                else:
                    string = f'[book] {news[1]["title"]}, description: {news[1]["description"]}\n'
            else:
                string = f'[book] {news[1]["title"]}\n'
            self._book_list.append((str(news[1]["bid"]), string))
        return self._book_list

    def get_book_dict(self):
        if self._book_dict is not None:
            return self._book_dict
        self._book_dict = {}
        for news in tqdm(self.book_df.iterrows()):
            bid = str(news[1]["bid"])
            desc = " ".join(news[1]["desc"].split(" ")[:50])
            if self.use_desc:
                self._book_dict[bid] = f'{news[1]["title"]}, description: {desc}'
            else:
                self._book_dict[bid] = news[1]["title"]
        return self._book_dict


class GoodreadsUser:
    def __init__(self, data_path, goodreads_prompter: GoodreadsPrompter):
        self.depot = UniDep(data_path, silent=True)
        self.bid = self.depot.vocabs("bid")
        self.book_dict = goodreads_prompter.get_book_dict()

        self._user_list = None

    def stringify(self):
        if self._user_list is not None:
            return self._user_list
        self._user_list = []
        for user in tqdm(self.depot):
            string = ""
            if not user["history"]:
                self._user_list.append((user["uid"], None))
            for i, n in enumerate(user["history"]):
                string += f"({i + 1}) {self.book_dict[self.bid.i2o[n]]}\n"
            self._user_list.append((user["uid"], string))
        return self._user_list


class GoodreadsColdUser:
    def __init__(self, data_path, goodreads_prompter: GoodreadsPrompter):
        self.depot = UniDep(data_path, silent=True)
        self.bid = self.depot.vocabs("bid")
        self.book = goodreads_prompter.get_book_dict()

        self._user_list = None

    def stringify(self):
        if self._user_list is not None:
            return self._user_list
        self._user_list = []
        for user in tqdm(self.depot):
            string = ""
            if not user["history"] or len(user["history"]) > 5:
                continue
            for i, n in enumerate(user["history"]):
                string += f"({i + 1}) {self.book[self.bid.i2o[n]]}\n"
            self._user_list.append((user["uid"], string))
        return self._user_list


class GoodreadsCoT:
    def __init__(self, data_path, profile_path, goodreads_prompter: GoodreadsPrompter):
        self.depot = UniDep(data_path, silent=True)

        self.topics = dict()
        with open(profile_path, "r") as f:
            for line in f:
                if line.endswith("\n"):
                    line = line[:-1]
                data = json.loads(line)
                uid, topic = data["uid"], data["interest"]
                self.topics[uid] = topic
        self.bid = self.depot.vocabs("bid")
        self.book_dict = goodreads_prompter.get_book_dict()

        self._user_list = None

    def stringify(self):
        if self._user_list is not None:
            return self._user_list
        self._user_list = []
        for user in tqdm(self.depot):
            string = ""
            pg = self.topics[user["uid"]]
            string += "Interest Topics:\n"
            for t in pg:
                string += f"- {t}\n"
            string += "\n"

            string += "History:\n"
            for i, n in enumerate(user["history"]):
                string += f"({i + 1}) {self.book_dict[self.bid.i2o[n]]}\n"
            self._user_list.append((user["uid"], string))
        return self._user_list
