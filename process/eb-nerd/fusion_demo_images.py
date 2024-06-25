from UniTok import UniDep
import numpy as np
import os

news = UniDep(os.path.join("ebnerd-benchmark/data/ebnerd_demo/tokenized_bert_images", 'news'))
news_llama = UniDep('ebnerd-benchmark/data/ebnerd_demo/tokenized_llama_images/news-llama')

news.rename_col('title', 'title-bert')
news.rename_col('subtitle', 'subtitle-bert')
news.rename_col('body', 'body-bert')
news.rename_col('category', 'category-token')
news.rename_col('image_caption_text', 'image-caption-text-token')

news_llama.rename_col('title', 'title-llama')
news_llama.rename_col('subtitle', 'subtitle-llama')
news_llama.rename_col('body', 'body-llama')
news_llama.rename_col('category', 'category-llama')
news_llama.rename_col('image_caption_text', 'image-caption-text-llama')

news.inject(news_llama, ['title-llama', 'subtitle-llama', 'body-llama', 'category-llama', 'image-caption-text-llama'])
news.export('ebnerd-benchmark/data/ebnerd_demo/news_fusion_images')