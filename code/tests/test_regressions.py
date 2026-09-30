"""Offline regression tests; pandas is required, GPU libraries are mocked."""

import importlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
import prompt_utils
import reasoning_utils

# Dataset inheritance does not affect the data transformations tested here.
torch_data = ModuleType("torch.utils.data")
torch_data.Dataset = object
with patch.dict(
    sys.modules,
    {
        "torch": ModuleType("torch"),
        "torch.utils": ModuleType("torch.utils"),
        "torch.utils.data": torch_data,
    },
):
    dataset_utils = importlib.import_module("dataset_utils")


class TextTokenizer:
    def encode(self, text, **kwargs):
        return list(text)

    def decode(self, tokens, **kwargs):
        return "".join(tokens)

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_options = kwargs
        return "rendered"

    def __call__(self, **kwargs):
        self.input_options = kwargs

        class Inputs(dict):
            def to(self, device):
                self.device = device
                return self

        return Inputs(input_ids=SimpleNamespace(shape=(1, 2)))


class ReasoningTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TextTokenizer()
        self.sample = {
            "question": "Why?",
            "answer": "Because",
            "content": "abcdef",
            "image": None,
        }

    def test_text_generation_uses_text_messages_and_model_device(self):
        model = SimpleNamespace(
            device="cpu", generate=MagicMock(return_value=[list("__answer")])
        )
        result = reasoning_utils.generate_response(
            "question", None, model, self.tokenizer
        )
        self.assertEqual(result, "answer")
        self.assertEqual(self.tokenizer.messages[0]["content"], "question")
        self.assertFalse(self.tokenizer.template_options["tokenize"])
        self.assertNotIn("images", self.tokenizer.input_options)

    def test_processor_only_adds_image_when_present(self):
        processor = TextTokenizer()
        processor.tokenizer = TextTokenizer()
        model = SimpleNamespace(
            device="cpu", generate=MagicMock(return_value=[list("__ok")])
        )
        reasoning_utils.generate_response("question", None, model, processor)
        self.assertEqual(
            [part["type"] for part in processor.messages[0]["content"]], ["text"]
        )
        picture = object()
        reasoning_utils.generate_response("question", picture, model, processor)
        self.assertIs(processor.input_options["images"], picture)
        self.assertEqual(processor.messages[0]["content"][0]["type"], "image")

    def test_plain_qa_context_is_read_by_memory_reasoning(self):
        with patch.object(
            reasoning_utils, "generate_response", side_effect=["memory", "answer"]
        ) as generate:
            output = reasoning_utils.mem_reasoning(self.sample, None, self.tokenizer, 3)
        self.assertEqual(output[:3], ("Why?", "Because", "answer"))
        self.assertIn("abcdef", generate.call_args_list[0].args[0])
        self.assertIn("memory", generate.call_args_list[1].args[0])

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            reasoning_utils.mem_reasoning(self.sample, None, self.tokenizer, 0)

    def test_recurrent_conversion_uses_new_chunks_and_previous_memory(self):
        with (
            patch.object(prompt_utils, "RECURRENT_CHUNK_SIZE", 3),
            patch.object(
                reasoning_utils,
                "generate_response",
                side_effect=["first memory", "second memory"],
            ) as generate,
        ):
            result = prompt_utils.convert_to_conversation_recurrent(
                self.sample, None, self.tokenizer
            )
        self.assertIn("abc", generate.call_args_list[0].args[0])
        self.assertIn("def", generate.call_args_list[1].args[0])
        self.assertIn("first memory", generate.call_args_list[1].args[0])
        self.assertIn("second memory", result["messages"][0]["content"])

    def test_head_tail_truncation_preserves_original_count(self):
        with patch.object(prompt_utils, "MAX_CONTEXT_LEN", 5):
            tokens, count = reasoning_utils._context_tokens("abcdefgh", self.tokenizer)
        self.assertEqual("".join(tokens), "abfgh")
        self.assertEqual(count, 8)

    def test_question_is_not_returned_as_a_list(self):
        self.sample["question"] = prompt_utils.SECTION_SEPARATOR.join(
            ("Candidates", "Remember", "Answer")
        )
        with patch.object(reasoning_utils, "generate_response", return_value="ok"):
            result = reasoning_utils.vanilla_reasoning(
                self.sample, None, self.tokenizer, None
            )
        self.assertEqual(result[0], self.sample["question"])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        metadata = [
            {
                "parent_asin": item,
                "title": item,
                "description": "description",
                "features": ["feature"],
            }
            for item in ("history", "target", "negative", "target2")
        ]
        pd.DataFrame(metadata).to_json(
            self.directory / "meta_Video_Games.jsonl.gz",
            orient="records",
            lines=True,
            compression="gzip",
        )
        pd.DataFrame(
            [{"parent_asin": "history", "user_id": "u1", "text": "review"}]
        ).to_json(
            self.directory / "Video_Games.jsonl.gz",
            orient="records",
            lines=True,
            compression="gzip",
        )
        (self.directory / "train_valid.txt").write_text("u1 history\nu2 history\n")
        # Test rows deliberately use a different order from training rows.
        (self.directory / "test.txt").write_text("u2 target2\nu1 target\n")
        (self.directory / "item_list.txt").write_text(
            "history\ntarget\nnegative\nnegative\nmissing\n"
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_headerless_data_keeps_first_user_and_joins_targets_by_user(self):
        data = dataset_utils.VideoGamesDataset(self.directory, task="searching")
        self.assertEqual(len(data), 2)
        sample = data[0]
        self.assertEqual(sample["answer"], "target")
        self.assertEqual(sample["candidates"].count("Title: target\n"), 1)
        self.assertNotIn("Title: history", sample["candidates"])
        self.assertNotIn("Title: missing", sample["candidates"])
        self.assertEqual(sample["candidates"].count("Title: negative\n"), 1)
        self.assertIn("Description: description", sample["content"])

    def test_judging_negatives_exclude_positive_items(self):
        data = dataset_utils.VideoGamesDataset(self.directory)
        with patch.object(random, "choice", side_effect=[False, "negative"]):
            sample = data[0]
        self.assertEqual(sample["answer"], "No")
        self.assertIn("Title: negative", sample["candidates"])

    def test_hotpot_sample_contract(self):
        path = self.directory / "qa.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "index": 1,
                        "input": "Why?",
                        "answers": ["Because"],
                        "context": "text",
                        "num_docs": 2,
                    }
                ]
            )
        )
        sample = dataset_utils.HotPotQADataset(path)[0]
        self.assertEqual(sample["candidates"], "")
        self.assertEqual(sample["seq_l"], 2)

    def test_instructrec_does_not_reveal_target_label(self):
        pd.DataFrame(
            [
                {
                    "title": ["history"],
                    "description": ["description"],
                    "reviewText": ["review"],
                    "persona": "reader",
                    "instruction": "suggest a book",
                    "ranked_lists": [1, 2],
                }
            ]
        ).to_pickle(self.directory / "booksAll_recagent.pkl")
        pd.DataFrame(
            [
                {"index": 1, "title": "book A", "description": "A"},
                {"index": 2, "title": "book B", "description": "B"},
            ]
        ).to_csv(self.directory / "combined_books_asin_mapping.csv", index=False)
        data = dataset_utils.InstructRecDataset(
            self.directory, domain="books", task="judging"
        )
        with patch.object(random, "choice", return_value=True):
            sample = data[0]
        self.assertEqual(sample["answer"], "Yes")
        self.assertNotIn("target", sample["question"])
        self.assertNotIn("negative", sample["question"])


class TrainingTests(unittest.TestCase):
    def test_train_is_called_before_save_and_grpo_receives_rewards(self):
        unsloth = ModuleType("unsloth")
        unsloth.FastLanguageModel = MagicMock()
        unsloth.FastVisionModel = MagicMock()
        collators = ModuleType("unsloth.trainer")
        collators.UnslothVisionDataCollator = MagicMock()
        trl = ModuleType("trl")
        for name in ("SFTConfig", "SFTTrainer", "GRPOConfig", "GRPOTrainer"):
            setattr(trl, name, MagicMock())
        with patch.dict(
            sys.modules, {"unsloth": unsloth, "unsloth.trainer": collators, "trl": trl}
        ):
            sys.modules.pop("learning_utils", None)
            learning = importlib.import_module("learning_utils")
            for method, trainer_class in (
                (learning.sft_tuning, trl.SFTTrainer),
                (learning.grpo_tuning, trl.GRPOTrainer),
            ):
                events = []
                model = SimpleNamespace(
                    save_pretrained=lambda path: events.append("save")
                )
                tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
                trainer_class.return_value.train.side_effect = lambda: events.append(
                    "train"
                )
                kwargs = (
                    {"reward_funcs": ["reward-model"]}
                    if method == learning.grpo_tuning
                    else {}
                )
                method(model, tokenizer, [], **kwargs)
                self.assertEqual(events, ["train", "save"])
            self.assertEqual(
                trl.GRPOTrainer.call_args.kwargs["reward_funcs"], ["reward-model"]
            )
            self.assertNotIn("data_collator", trl.GRPOTrainer.call_args.kwargs)

    def test_checkpoint_path_and_modality_are_respected(self):
        unsloth = ModuleType("unsloth")
        unsloth.FastLanguageModel = MagicMock()
        unsloth.FastVisionModel = MagicMock()
        unsloth.FastLanguageModel.from_pretrained.return_value = ("model", "tokenizer")
        with patch.dict(sys.modules, {"unsloth": unsloth}):
            sys.modules.pop("model_utils", None)
            module = importlib.import_module("model_utils")
            result = module.load_model(
                SimpleNamespace(
                    modal="text",
                    load_checkpoint="custom-checkpoint",
                    model="base",
                    hf_token=None,
                )
            )
        self.assertEqual(result, ("model", "tokenizer"))
        self.assertEqual(
            unsloth.FastLanguageModel.from_pretrained.call_args.kwargs["model_name"],
            "custom-checkpoint",
        )
        unsloth.FastVisionModel.from_pretrained.assert_not_called()
        unsloth.FastLanguageModel.get_peft_model.assert_not_called()


class JudgeTests(unittest.TestCase):
    def test_invalid_responses_are_errors_and_valid_responses_are_normalized(self):
        openai = ModuleType("openai")
        openai.OpenAI = MagicMock()
        dotenv = ModuleType("dotenv")
        dotenv.load_dotenv = MagicMock()
        with patch.dict(sys.modules, {"openai": openai, "dotenv": dotenv}):
            sys.modules.pop("evaluation_utils", None)
            evaluation = importlib.import_module("evaluation_utils")
            for body, expected in (
                ("[]", "error"),
                ('{"judgement": null}', "error"),
                ('{"judgement": " Correct "}', "correct"),
            ):
                openai.OpenAI.return_value.chat.completions.create.return_value = (
                    SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=body))]
                    )
                )
                with patch.object(evaluation.logger, "warning"):
                    result = evaluation.llm_as_judge("question", "answer", "prediction")
                self.assertEqual(result["judgement"], expected)


class MainTests(unittest.TestCase):
    def test_qa_evaluation_logs_judge_errors_without_counting_them_as_wrong(self):
        sample = {
            "idx": 1,
            "question": "Why?",
            "answer": "Because",
            "content": "context",
        }
        modules = {}
        for name in (
            "unsloth",
            "torch",
            "dataset_utils",
            "evaluation_utils",
            "model_utils",
            "reasoning_utils",
        ):
            modules[name] = MagicMock()
        modules["dataset_utils"].load_dataset.return_value = [sample, sample]
        modules["model_utils"].load_model.return_value = ("model", "tokenizer")
        modules["reasoning_utils"].vanilla_reasoning.return_value = (
            "Why?",
            "Because",
            "Because",
            7,
        )
        modules["evaluation_utils"].llm_as_judge.side_effect = [
            {"judgement": "error"},
            {"judgement": "correct"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            fake_file = str(Path(directory) / "code" / "main.py")
            with (
                patch.dict(sys.modules, modules),
                patch.object(main, "__file__", fake_file),
                patch("builtins.print"),
            ):
                main.main(
                    [
                        "--dataset",
                        "hotpotqa",
                        "--reasoning",
                        "vanilla",
                        "--modal",
                        "text",
                    ]
                )
            path = next((Path(directory) / "result").glob("*.jsonl"))
            logs = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertIsNone(logs[0]["current_accuracy"])
        self.assertEqual(logs[1]["current_accuracy"], 1)
        self.assertEqual(logs[1]["judge_errors"], 1)
        self.assertEqual(logs[1]["current_average_input_length"], 11)

    def test_empty_dataset_does_not_load_model_or_divide_by_zero(self):
        modules = {
            name: MagicMock()
            for name in (
                "unsloth",
                "torch",
                "dataset_utils",
                "evaluation_utils",
                "model_utils",
                "reasoning_utils",
            )
        }
        modules["dataset_utils"].load_dataset.return_value = []
        with patch.dict(sys.modules, modules), patch("builtins.print"):
            main.main([])
        modules["model_utils"].load_model.assert_not_called()


class UtilityTests(unittest.TestCase):
    def test_cli_rejects_invalid_chunk_size(self):
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            main.parse_args(["--chunk_size", "0"])

    def test_timer_can_restart(self):
        import utils

        timer = utils.TaskTimer()
        with patch.object(utils.time, "perf_counter", side_effect=[1, 3, 10, 14]):
            timer.start()
            self.assertEqual(timer.stop(), 2)
            timer.start()
            self.assertEqual(timer.elapsed(), 4)


if __name__ == "__main__":
    unittest.main()
