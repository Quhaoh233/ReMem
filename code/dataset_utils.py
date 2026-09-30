"""Adapt local datasets to the common question/content/answer sample format.

Recommendation questions contain three fields separated by SECTION_SEPARATOR;
content uses the same delimiter between history items. QA questions are plain text.
"""

import json
import pickle
import random
from pathlib import Path

import pandas as pd
import prompt_utils
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RECOMMENDATION_TASKS = {"searching", "judging"}


def load_dataset(args):
    """Load supported evaluation data relative to the project, not the shell."""
    if args.dataset == "webwalkerqa_main":
        return WebWalkerDataset()
    if args.dataset in {"movietv", "books"}:
        return InstructRecDataset(domain=args.dataset, task=args.task)
    if args.dataset == "games":
        return VideoGamesDataset(task=args.task)
    if args.dataset == "hotpotqa":
        return HotPotQADataset()
    # Raw Mind2Web rows do not implement this project's sample contract.
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def _question(candidates, task, instruction=""):
    """Join candidate information, memory instructions, and the final question."""
    memory_question = "Based on the user's interaction history, summarize preferences that can help the recommendation."
    if instruction:
        memory_question += f" Recommendation instruction: [{instruction}]"
    final_question = (
        prompt_utils.SEARCHING_QUESTION
        if task == "searching"
        else prompt_utils.JUDGING_QUESTION
    )
    # split_question() depends on this order: candidates, memory task, final task.
    return prompt_utils.SECTION_SEPARATOR.join(
        (candidates, memory_question, final_question)
    )


def _as_text(value):
    """Metadata descriptions may be lists, strings, or missing values."""
    if isinstance(value, (list, tuple)):
        return " ".join(str(part) for part in value)
    return "unknown" if value is None or pd.isna(value) else str(value)


class WebWalkerDataset(Dataset):
    """Read cached OCR text for web questions; screenshots are not model inputs."""

    def __init__(self, file_path=None):
        self.file_path = (
            Path(file_path)
            if file_path is not None
            else DATA_DIR / "webwalkerqa_main.jsonl"
        )
        self.data = pd.read_json(self.file_path, lines=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        info = row.get("info", {})
        if not isinstance(info, dict):
            info = {}
        ocr_path = (
            self.file_path.parent
            / "webwalkerqa"
            / "ocr_results"
            / f"website_{idx}"
            / "result.mmd"
        )
        try:
            ocr_content = ocr_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            ocr_content = "No content available."
        return {
            "idx": idx,
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "content": f"The parsing of the website screenshot is as follows:\n{ocr_content}",
            "root_url": row.get("root_url", ""),
            "domain": info.get("domain", ""),
            "difficulty": info.get("difficulty_level", ""),
            "lang": info.get("lang", "en"),
            "question_type": info.get("type", ""),
            "candidates": "",
            "image": None,
            "seq_l": 1,
        }


class InstructRecDataset(Dataset):
    """Build recommendation examples from trusted local InstructRec pickle files."""

    def __init__(self, file_path=None, domain="movietv", task="searching"):
        if domain not in {"movietv", "books"}:
            raise ValueError(f"Unsupported domain: {domain}")
        if task not in RECOMMENDATION_TASKS:
            raise ValueError(f"Unsupported task: {task}")
        directory = Path(file_path) if file_path is not None else DATA_DIR / domain
        with (directory / f"{domain}All_recagent.pkl").open("rb") as file:
            self.data = pickle.load(file)
        self.asin_mapping = pd.read_csv(
            directory / f"combined_{domain}_asin_mapping.csv"
        )
        self.domain = domain
        self.task = task

    def __len__(self):
        return len(self.data)

    def _candidate(self, item_id):
        rows = self.asin_mapping[self.asin_mapping["index"] == item_id]
        if rows.empty:
            raise ValueError(f"Missing candidate metadata for item {item_id!r}")
        # Mapping files can repeat IDs; preserve the original first-match policy.
        return rows.iloc[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        titles = row["title"]
        content = (
            f"You have some information about this user: {row['persona']}.\n"
            "The user's historical interacted items are in chronological order:\n"
        )
        # Keep one delimiter per interaction so memory reasoning can group items.
        for number, title in enumerate(titles):
            content += (
                f"{prompt_utils.SECTION_SEPARATOR}\nInteracted item {number + 1}:\n"
                f"Title: {title}\nDescription: {_as_text(row['description'][number])}\n"
                f"Review Text: {_as_text(row['reviewText'][number])}\n"
            )
        ranked_items = list(row["ranked_lists"])
        # Some exports contain multiple candidate lists. Use the first list.
        if ranked_items and isinstance(ranked_items[0], (list, tuple)):
            ranked_items = list(ranked_items[0])
        if not ranked_items:
            raise ValueError(f"No candidates for sample {idx}")
        # The source export puts the correct item first, before prompt shuffling.
        target_id = ranked_items[0]
        if self.task == "searching":
            golden_answer = self._candidate(target_id)["title"]
            candidate_ids = ranked_items.copy()
            # Prevent candidate position from revealing the expected answer.
            random.shuffle(candidate_ids)
        else:
            negative_ids = [item for item in ranked_items[1:] if item != target_id]
            if not negative_ids:
                raise ValueError(f"No negative candidates for sample {idx}")
            # Judging presents one item and uses its sampled class as Yes/No truth.
            is_target = random.choice((True, False))
            candidate_ids = [target_id if is_target else random.choice(negative_ids)]
            golden_answer = "Yes" if is_target else "No"
        candidates = "Candidate items:\n"
        for number, item_id in enumerate(candidate_ids, start=1):
            candidate = self._candidate(item_id)
            # Neutral numbering avoids leaking target/negative labels to the model.
            candidates += (
                f"Candidate {number}:\nTitle: {candidate['title']}\n"
                f"Description: {_as_text(candidate['description'])}\n"
            )
        return {
            "idx": idx,
            "question": _question(candidates, self.task, row["instruction"]),
            "answer": golden_answer,
            "content": content,
            "domain": self.domain,
            "candidates": candidates,
            "image": None,
            "seq_l": len(titles),
        }


class VideoGamesDataset(Dataset):
    """Build examples from headerless user/item interaction files and metadata."""

    def __init__(self, file_path=None, task="judging", item_num_threshold=50):
        if task not in RECOMMENDATION_TASKS:
            raise ValueError(f"Unsupported task: {task}")
        if item_num_threshold <= 0:
            raise ValueError("item_num_threshold must be positive.")
        directory = Path(file_path) if file_path is not None else DATA_DIR / "games"
        self.meta_data = pd.read_json(
            directory / "meta_Video_Games.jsonl.gz", lines=True
        )
        self.reviews = pd.read_json(directory / "Video_Games.jsonl.gz", lines=True)
        # Each line is "user_id item_id ..."; the first line is data, not a header.
        self.train_valid = pd.read_csv(
            directory / "train_valid.txt", header=None, dtype=str
        )
        self.test = pd.read_csv(directory / "test.txt", header=None, dtype=str)
        self.item_list = pd.read_csv(
            directory / "item_list.txt", header=None, dtype=str
        )[0].tolist()
        self.task = task
        self.item_num_threshold = item_num_threshold
        self._metadata = self.meta_data.drop_duplicates("parent_asin").set_index(
            "parent_asin"
        )
        # Join by user ID because train and test rows may have different orders.
        self._targets = {}
        for line in self.test[0]:
            user, *items = line.split()
            if not items:
                raise ValueError(f"Missing test item for user {user}")
            self._targets[user] = items[0]
        # Preserve source order while removing duplicates and missing metadata.
        self._candidate_items = list(
            dict.fromkeys(
                item for item in self.item_list if item in self._metadata.index
            )
        )

    def __len__(self):
        return len(self.train_valid)

    def _item_text(self, item):
        if item not in self._metadata.index:
            return "unknown", "unknown", "unknown"
        info = self._metadata.loc[item]
        return tuple(
            _as_text(info[field]) for field in ("title", "description", "features")
        )

    def __getitem__(self, idx):
        user, *all_items = self.train_valid.iloc[idx, 0].split()
        # Show recent history, but retain all positives for negative filtering.
        items = all_items[-self.item_num_threshold :]
        target = self._targets.get(user)
        if target is None or target not in self._metadata.index:
            raise ValueError(f"Missing test target or metadata for user {user}")
        # Exclude all historical positives and the held-out target from negatives.
        excluded = set(all_items) | {target}
        negative_pool = [item for item in self._candidate_items if item not in excluded]
        if not negative_pool:
            raise ValueError(f"No negative candidates for user {user}")
        content = "The user's historical interacted items are in chronological order:\n"
        for number, item in enumerate(items, start=1):
            reviews = self.reviews.loc[
                (self.reviews["parent_asin"] == item)
                & (self.reviews["user_id"] == user),
                "text",
            ]
            review = reviews.iloc[0] if not reviews.empty else "No review available."
            title, description, features = self._item_text(item)
            content += (
                f"{prompt_utils.SECTION_SEPARATOR}\nInteracted item {number}:\n"
                f"Title: {title}\nDescription: {description}\n"
                f"Features: {features}\nReview: {review}\n"
            )
        if self.task == "searching":
            golden_answer = self._item_text(target)[0]
            candidate_ids = [target] + random.sample(
                negative_pool, min(9, len(negative_pool))
            )
            # Prevent candidate position from revealing the expected answer.
            random.shuffle(candidate_ids)
        else:
            # Judging presents one item and uses its sampled class as Yes/No truth.
            is_target = random.choice((True, False))
            candidate_ids = [target if is_target else random.choice(negative_pool)]
            golden_answer = "Yes" if is_target else "No"
        candidates = "Candidate items:\n"
        for number, item in enumerate(candidate_ids, start=1):
            title, description, features = self._item_text(item)
            candidates += (
                f"Candidate {number}:\nTitle: {title}\n"
                f"Description: {description}\nFeatures: {features}\n"
            )
        return {
            "idx": idx,
            "question": _question(candidates, self.task),
            "answer": golden_answer,
            "content": content,
            "domain": "videogames",
            "candidates": candidates,
            "image": None,
            # Report the original history length, before the display threshold.
            "seq_l": len(all_items),
        }


class HotPotQADataset(Dataset):
    """Read long-context QA examples using the same fields as recommendation data."""

    def __init__(self, file_path=None):
        path = (
            Path(file_path)
            if file_path is not None
            else DATA_DIR / "hotpotqa" / "eval_400.json"
        )
        with path.open("r", encoding="utf-8") as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "idx": sample["index"],
            "question": sample["input"],
            "answer": sample["answers"],
            "content": sample["context"],
            "domain": "hotpotqa",
            "candidates": "",
            "image": None,
            "seq_l": sample.get("num_docs", 0),
        }
