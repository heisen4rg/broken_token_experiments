import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hf_cache  # noqa: E402, F401  (sets HF_HOME before transformers is imported)

from transformers import AutoTokenizer  # noqa: E402
import random
import string


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
        if char in vocab:
            l.append(char)
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


def bpe_dropout_segment(text: str, tokenizer, p: float) -> list[str]:
    """
    Tokenizes `text` with BPE-dropout (Provilkov et al.), drop probability `p`.

    p=0.0 reproduces canonical tokenization; p=1.0 degenerates to per-character
    tokens. Operates on raw text (not a single canonical token) since the
    tokenizer's own pre-tokenizer must run first to establish word boundaries.

    Note: mutates the tokenizer's backend BPE model for the duration of the
    call, so it is not safe to call concurrently on the same tokenizer object
    from multiple threads.
    """
    backend_model = tokenizer.backend_tokenizer.model
    original_dropout = backend_model.dropout
    backend_model.dropout = p
    try:
        return tokenizer.tokenize(text)
    finally:
        backend_model.dropout = original_dropout


def misspell_word(word: str) -> str:
    """
    Randomly adds, removes, or substitutes a single character in `word`,
    matching the misspelling procedure in Appendix B.5.
    """
    operations = ["add", "remove", "substitute"]
    if len(word) < 2:
        operations.remove("remove")

    operation = random.choice(operations)

    if operation == "add":
        pos = random.randint(0, len(word))
        char = random.choice(string.ascii_lowercase)
        return word[:pos] + char + word[pos:]

    if operation == "remove":
        pos = random.randrange(len(word))
        return word[:pos] + word[pos + 1:]

    # substitute
    pos = random.randrange(len(word))
    char = random.choice([c for c in string.ascii_lowercase if c != word[pos]])
    return word[:pos] + char + word[pos + 1:]


def tokenize_non_canonical(text: str, tokenizer, segment_fn) -> list[int]:
    """
    Tokenizes `text` non-canonically and returns token IDs ready for a model.

    `segment_fn` is `build_segments` or `character_segment` (both share the
    `(token: str, vocab: set[str]) -> list[str]` signature): each canonical
    token of `text` is re-segmented independently and the resulting pieces
    are concatenated, then converted to token IDs.

    Special/added tokens (e.g. chat-template control tokens like
    <|begin_of_text|>) are passed through unsplit -- they aren't text the
    BPE vocabulary encodes, so resegmenting their literal string would shred
    them into meaningless byte pieces (e.g. '<|begin_of_text|>' -> '<', '|',
    'b', ...) instead of leaving them as the single control token they are.
    """
    vocab = set(tokenizer.get_vocab().keys())
    special_tokens = set(tokenizer.all_special_tokens) | set(tokenizer.get_added_vocab().keys())
    canonical_tokens = tokenizer.tokenize(text)

    non_canonical_tokens = []
    for token in canonical_tokens:
        if token in special_tokens:
            non_canonical_tokens.append(token)
        else:
            non_canonical_tokens.extend(segment_fn(token, vocab))

    return tokenizer.convert_tokens_to_ids(non_canonical_tokens)


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct"
    )

    vocab = set(tokenizer.get_vocab().keys())

    for c in "abcdefghijklmnopqrstuvwxyz0123456789":
        if c not in vocab:
            print(c)

    word = " unbelievable"

    canonical = tokenizer.tokenize(word)
    assert bpe_dropout_segment(word, tokenizer, 0.0) == canonical
    assert tokenizer.tokenize(word) == canonical  # dropout was reset

    full_char = bpe_dropout_segment(word, tokenizer, 1.0)
    assert all(t in vocab for t in full_char)
    assert len(full_char) == len(word)

    outputs = {tuple(bpe_dropout_segment(word, tokenizer, 0.5)) for _ in range(20)}
    assert len(outputs) > 1
    assert all(t in vocab for out in outputs for t in out)

    print("bpe_dropout_segment checks passed")

    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "Describe Seattle's weather.<|eot_id|>"
    )

    canonical_ids = tokenizer.encode(prompt, add_special_tokens=False)

    char_ids = tokenize_non_canonical(prompt, tokenizer, character_segment)
    assert tokenizer.decode(char_ids) == tokenizer.decode(canonical_ids)
    assert len(char_ids) >= len(canonical_ids)
    # special tokens must survive as single ids, not get shredded
    for special in tokenizer.all_special_tokens:
        special_id = tokenizer.convert_tokens_to_ids(special)
        assert char_ids.count(special_id) == canonical_ids.count(special_id)

    random_ids = tokenize_non_canonical(prompt, tokenizer, build_segments)
    assert tokenizer.decode(random_ids) == tokenizer.decode(canonical_ids)
    for special in tokenizer.all_special_tokens:
        special_id = tokenizer.convert_tokens_to_ids(special)
        assert random_ids.count(special_id) == canonical_ids.count(special_id)

    print("tokenize_non_canonical checks passed")

    for base_word in ["it", "a", "cat", "seattle", "unbelievable"]:
        for _ in range(30):
            result = misspell_word(base_word)
            assert result != base_word
            assert abs(len(result) - len(base_word)) <= 1
            if len(base_word) < 2:
                assert len(result) >= len(base_word)  # "remove" excluded, can't go to 0

    print("misspell_word checks passed")
