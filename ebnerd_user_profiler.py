import json
import time
import os

from tqdm import tqdm
import pandas as pd

from processor.ebnerd.prompter import EbnerdPrompter, EbnerdUser
from utils.openai.chat_service import ChatService

MIN_INTERVAL = 0

ebnerd_prompter = EbnerdPrompter("data/eb-nerd/ebnerd_small/articles.parquet")

articles_path = "data/eb-nerd/ebnerd_small/articles.parquet"
history_path = "data/eb-nerd/ebnerd_small/train/history.parquet"
behaviors_path = "data/eb-nerd/ebnerd_small/train/behaviors.parquet"

user_list = EbnerdUser(history_path, behaviors_path, ebnerd_prompter).stringify()

system = """You are asked to describe user interest based on his/her characteristics (if known) and top 100 browsed news list, the format of which is as below:

[gender] {gender}
[postcode] {postcode}
[age] {age}
(1) title: {news_title}, read time: {read_time}, scroll percentage: {scroll_percentage}
...
(n) title: {news_title}, read time: {read_time}, scroll percentage: {scroll_percentage}

You can only response the user interests with the following format to describe the [topics] and [regions] of the user's interest

[topics]
- topic1
- topic2
...

where topic is limited to the following options: 

(1) crime and safety
(2) entertainment
(3) media
(4) finance
(5) sports
(6) business
(7) politics
(8) health and well-being
(9) transport
(10) science and technology
(11) education
(12) conflict and disasters
(13) lifestyle and society
(14) travelling
(15) nature and environment
(16) housing and real estate
(17) art and culture
(18) events

When recommending topics, keep in mind that the top 100 articles is ordered based on the user's reading time, so give more importance to the top articles.
Only [topics] can appear in your response. Your response topic list should be ordered, that the first several options should be most related to the user's interest. You are not allowed to respond with any other words, explanations or notes. Now, the task formally begins. Any other information should not disturb you."""

save_path = "data/eb-nerd/user_profiler.log"

# Create the file if it doesn't exist
if not os.path.exists(save_path):
    with open(save_path, "w"):
        pass  

exist_set = set()
with open(save_path, "r") as f:
    for line in f:
        data = json.loads(line)
        exist_set.add(data["user_id"])

empty_count = 0

for uid, content in tqdm(user_list):
    start_time = time.time()
    if uid in exist_set:
        continue

    if not content:
        empty_count += 1
        continue

    try:
        service = ChatService(system)
        enhanced = service.ask(content)  # type: str
        enhanced = enhanced.rstrip("\n")

        with open(save_path, "a") as f:
            f.write(json.dumps({"user_id": uid, "interest": enhanced}) + "\n")
    except Exception as e:
        print(e)

    interval = time.time() - start_time
    if interval <= MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - interval)

print("empty count: ", empty_count)
