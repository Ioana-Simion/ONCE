import time
import os

from tqdm import tqdm

from utils.openai.chat_service import ChatService

from processor.ebnerd.prompter import EbnerdPrompter

MIN_INTERVAL = 1.5

# concise

news_list = EbnerdPrompter("data/eb-nerd/ebnerd_small/articles.parquet").stringify()[:5]

system = """You are asked to act as a news content summarizer. I will provide you a piece of news, with its original title, subtitle, category, topics and body. The news format is as below:

[title] {title}
[subtitle] {subtitle}
[category] {category}
[topics] {topics}
[body] {body}

where {title}, {subtitle}, {category}, {topics} and {body} will be filled with content. You can only respond with a summarized body content which shold be clear, complete, objective and neutral. You are not allowed to response any other words for any explanation. Your response format should be:

[summarized_body] {summarized_body}

where {summarized_body} should be filled with the summarized body. Now, your role of a news content summarizer formally begins. Any other information should not disturb your role."""


save_path = "data/eb-nerd/ebnerd_news_summarizer.log"

# Create the file if it doesn't exist
if not os.path.exists(save_path):
    with open(save_path, "w"):
        pass  

exist_set = set()
with open(save_path, "r") as f:
    for line in f:
        if line and line.startswith("N"):
            exist_set.add(line.split("\t")[0])


for article_id, content in tqdm(news_list):
    start_time = time.time()
    if article_id in exist_set:
        continue

    try:
        service = ChatService(system)
        enhanced = service.ask(content)
        enhanced = enhanced.rstrip("\n")

        with open(save_path, "a") as f:
            f.write(f"{article_id}\t{enhanced}\n")
    except Exception as e:
        print(e)

    interval = time.time() - start_time
    if interval <= MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - interval)
