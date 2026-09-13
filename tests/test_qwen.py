import torch

from src.models.qwen import GeneratorResult, QwenModel


def test_generator_result_contract():
    result = GeneratorResult(
        text="Paris",
        token_ids=[1001, 1002],
        tokens=["Paris", " "],
        probabilities=[0.98, 0.95],
    )

    assert isinstance(result.text, str)
    assert len(result.token_ids) == len(result.tokens)
    assert len(result.tokens) == len(result.probabilities)

    assert result.token_ids == [1001, 1002]
    assert result.tokens == ["Paris", " "]

    for probability in result.probabilities:
        assert 0.0 <= probability <= 1.0


def test_generator_result_optional_fields():
    result = GeneratorResult(
        text="Paris",
        token_ids=[1001],
        tokens=["Paris"],
        probabilities=[0.98],
    )

    assert result.offsets is None
    assert result.finish_reason is None


def test_generated_token_probability_is_selected_token_probability():
    result = GeneratorResult(
        text="Paris",
        token_ids=[1001, 1002],
        tokens=["Paris", " is"],
        probabilities=[0.90, 0.75],
    )

    assert result.probabilities[0] == 0.90
    assert result.probabilities[1] == 0.75
    assert len(result.probabilities) == len(result.token_ids)


class FakeTensor:
    def __init__(self, tensor):
        self.tensor = tensor

    def to(self, device):
        return self.tensor


class FakeTokenizer:
    def __call__(self, prompt, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[10, 11]]),
        }

    def decode(self, token_ids, skip_special_tokens=False):
        ids = [
            token_id.item() if hasattr(token_id, "item") else token_id
            for token_id in token_ids
        ]

        return "".join(f"token{token_id}" for token_id in ids)


class FakeModel:
    device = torch.device("cpu")

    def generate(self, **kwargs):
        # Input tokens: [10, 11]
        # Generated tokens: [2, 3]
        sequences = torch.tensor([[10, 11, 2, 3]])

        # Vocabulary size = 5
        #
        # For token 2:
        # logits = [0, 0, 0, 0, 0]
        #
        # For token 3:
        # logits = [0, 0, 0, 2, 0]
        #
        # Therefore token 3 should receive the highest probability.
        scores = (
            torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0, 2.0, 0.0]]),
        )

        class Output:
            pass

        output = Output()
        output.sequences = sequences
        output.scores = scores

        return output


def test_qwen_generate_extracts_selected_token_probabilities():
    tokenizer = FakeTokenizer()
    model = FakeModel()

    generator = QwenModel(
        tokenizer=tokenizer,
        model=model,
    )

    result = generator.generate(
        "What is the capital of France?",
        max_new_tokens=2,
        do_sample=False,
    )

    assert result.token_ids == [2, 3]

    assert result.tokens == [
        "token2",
        "token3",
    ]

    assert result.text == "token2token3"

    assert len(result.probabilities) == 2

    # First score tensor has equal logits, so every token has
    # probability 1/5.
    assert abs(result.probabilities[0] - 0.2) < 1e-6

    # Token 3 has the highest logit in the second score tensor.
    assert result.probabilities[1] > 0.5
