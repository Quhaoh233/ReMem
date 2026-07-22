import pandas as pd
import numpy as np
import sys


full = pd.read_csv("full.txt", header=None)[0].tolist()
# download the Video_Games.jsonl.gz file at https://amazon-reviews-2023.github.io/
reviews = pd.read_json("Video_Games.jsonl.gz", lines=True)


train = []
valid = []
test = []
train_valid = []
user_list = []
item_list = []
interaction_count = 0
for sample in full:
    print(sample)
    temp = sample.split(" ")
    user = temp[0]
    user_list.append(user)
    items = temp[1:]
    positive_items = []
    for item in items:
        item_list.append(item)
        rating = reviews.loc[(reviews["parent_asin"] == item) & (reviews["user_id"] == user), "rating"]
        if not rating.empty:
            rating = rating.values[0]
        else:
            rating = 1
        
        if rating > 3:
            positive_items.append(item)

    if len(positive_items) >= 5:
        interaction_count += len(positive_items)
        train_valid.append(" ".join([user]+positive_items[:-1]))
        train.append(" ".join([user]+positive_items[:-2]))
        valid.append(" ".join([user]+positive_items[-2:-1]))
        test.append(" ".join([user]+positive_items[-1:]))


with open("train_valid.txt", "w") as f:
    for line in train_valid:
        f.write(line + "\n")

with open("train.txt", "w") as f:
    for line in train:
        f.write(line + "\n")

with open("valid.txt", "w") as f:
    for line in valid:
        f.write(line + "\n")

with open("test.txt", "w") as f:
    for line in test:
        f.write(line + "\n")

user_list = list(set(user_list))
item_list = list(set(item_list))
# save the user and item list to a file
with open("user_list.txt", "w") as f:
    for user in user_list:
        f.write(user + "\n")

with open("item_list.txt", "w") as f:
    for item in item_list:
        f.write(item + "\n")

print(f"Number of users: {len(user_list)}")
print(f"Number of items: {len(item_list)}")
print(f"Number of interactions: {interaction_count}")

meta_data = pd.read_json("meta_Video_Games.jsonl.gz", lines=True)