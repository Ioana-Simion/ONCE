from UniTok import UniDep
import numpy as np
import os

news = UniDep(os.path.join("ebnerd-benchmark/data/tokenized_bert", 'news'))
news_llama = UniDep('ebnerd-benchmark/data/tokenized_llama/news-llama')

news.rename_col('title', 'title-bert')
news.rename_col('subtitle', 'subtitle-bert')
news.rename_col('body', 'body-bert')
news.rename_col('category', 'category-token')

news_llama.rename_col('title', 'title-llama')
news_llama.rename_col('subtitle', 'subtitle-llama')
news_llama.rename_col('body', 'body-llama')
news_llama.rename_col('category', 'category-llama')

news.inject(news_llama, ['title-llama', 'subtitle-llama', 'body-llama', 'category-llama'])
news.export('ebnerd-benchmark/data/news_fusion')