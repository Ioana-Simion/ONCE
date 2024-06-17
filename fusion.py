from UniTok import UniDep
import numpy as np
import os

# # Function to convert column data to numpy array and ensure homogeneous shape
# def convert_to_numpy_array_with_padding(data, column):
#     # Find the maximum length of the sequences in the column
#     max_length = max(len(x) if isinstance(x, (list, np.ndarray)) else 1 for x in data[column])
#     # Convert the column to a NumPy array with padding
#     padded_array = np.array([
#         np.pad(x, (0, max_length - len(x)), 'constant') if isinstance(x, (list, np.ndarray)) else np.pad([x], (0, max_length - 1), 'constant')
#         for x in data[column]
#     ])
#     return padded_array


news = UniDep(os.path.join("ebnerd-benchmark/data/tokenized_bert_test", 'news'))
news_llama = UniDep('ebnerd-benchmark/data/tokenized_llama_test/news-llama')

news.rename_col('title', 'title-bert')
news.rename_col('subtitle', 'subtitle-bert')
news.rename_col('body', 'body-bert')
news.rename_col('category', 'category-token')

news_llama.rename_col('title', 'title-llama')
news_llama.rename_col('subtitle', 'subtitle-llama')
news_llama.rename_col('body', 'body-llama')
news_llama.rename_col('category', 'category-llama')

news.inject(news_llama, ['title-llama', 'subtitle-llama', 'body-llama', 'category-llama'])
news.export('ebnerd-benchmark/data/news-fusion')


# # Convert columns to numpy arrays with padding
# columns_to_convert = ['title-llama', 'subtitle-llama', 'body-llama', 'category-llama']

# for col in columns_to_convert:
#     news_llama.data[col] = convert_to_numpy_array_with_padding(news_llama.data, col)

# # Inject llama data into bert data
# news.inject(news_llama, columns_to_convert)

# # Ensure all columns are numpy arrays with padding in the news object as well
# columns_to_convert_news = ['title-bert', 'subtitle-bert', 'body-bert', 'category-token'] + columns_to_convert

# for col in columns_to_convert_news:
#     news.data[col] = convert_to_numpy_array_with_padding(news.data, col)

# # Export the combined data
# news.export('ebnerd-benchmark/data/news-fusion2')