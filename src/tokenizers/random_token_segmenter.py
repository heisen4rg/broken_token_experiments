from functools import lru_cache
from transformers import AutoTokenizer
import random


def get_counter(token: str, vocab: set[str]):

    """

    Returns a memoized function count_from(start) that computes the number

    of valid segmentations beginning at index `start`.

    """

    @lru_cache(maxsize=None)

    def count_from(start: int) -> int:

        # Successfully segmented the entire token

        if start == len(token):

            return 1

        total = 0

        for end in range(start + 1, len(token) + 1):

            prefix = token[start:end]

            if prefix in vocab:

                total += count_from(end)

        return total

    return count_from


def count_segments(token: str, vocab: set[str]) -> int:

    count_from = get_counter(token, vocab)

    return count_from(0)


def build_segments(token: str, vocab: set[str]) -> list[str]:
    """
    Randomly samples one valid segmentation of a canonical token.

    Every complete segmentation has equal probability.
    """

    count_from = get_counter(token, vocab)

    segments = []
    start = 0

    while start < len(token):

        candidates = []
        weights = []

        # Find every valid prefix beginning at `start`
        for end in range(start + 1, len(token) + 1):

            prefix = token[start:end]

            if prefix in vocab:

                candidates.append((prefix, end))
                weights.append(count_from(end))

        # Randomly choose one branch using the subtree sizes
        chosen_prefix, next_start = random.choices(
            candidates,
            weights=weights,
            k=1
        )[0]

        segments.append(chosen_prefix)
        start = next_start

    return segments


def character_segment(token: str, vocab: set[str]) -> list[str]:
    l = []
    for char in token:
        if token in vocab:
            l.append(token)
        else:
            return []

    return l


def right_to_left_digit_segment(number: str) -> list[str]:
    assert number.isdigit(), "Input must contain only digits."
    segments = []

    i = len(number)

    while i > 0:

        start = max(0, i - 3)

        segments.append(number[start:i])

        i -= 3

    return segments[::-1]


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct"
    )

    vocab = set(tokenizer.get_vocab().keys())

    for c in "abcdefghijklmnopqrstuvwxyz0123456789":
        if c not in vocab:
            print(c)
