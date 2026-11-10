"""Evaluation metrics must measure what their names claim."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import (  # noqa: E402
    OtherCpuLoad,
    training_info,
    auto_flags,
    distinct_n,
    grouped_distinct,
    real_story_reference,
    repeated_ngram_rate,
    write_error_csv,
)
from src.utils import memory_mb, peak_memory_mb  # noqa: E402

LOOP = "The monster was scared and did not know what to do. " * 6
NORMAL = ("Lily said hello to Tom. Tom said hello to Ben. Ben said hello to Sue. "
          "Then they went to the park and played on the swing until the sun went down.")


def sample(completion, ended=True):
    return {"completion": completion, "text": "Once " + completion, "ended_with_eos": ended}


def test_common_phrase_is_not_flagged_but_a_loop_is():
    threshold = 0.2
    assert not auto_flags(sample(NORMAL), threshold)["repetition"]  # "said hello to" appears 3 times
    assert auto_flags(sample(LOOP), threshold)["repetition"]
    assert repeated_ngram_rate(LOOP) > repeated_ngram_rate(NORMAL)


def test_token_limit_is_separate_from_a_bad_ending():
    cut = auto_flags(sample("and then they", ended=False), 0.2)
    assert cut["incomplete"] and cut["hit_token_limit"]
    bad = auto_flags(sample("and then they", ended=True), 0.2)
    assert bad["incomplete"] and not bad["hit_token_limit"]
    good = auto_flags(sample("They went home.", ended=True), 0.2)
    assert not good["incomplete"] and not good["hit_token_limit"]


def test_distinct_compares_equal_lengths_and_groups():
    assert distinct_n(["a b c d"], 1) == 1.0
    assert distinct_n(["a a a b b b"], 1, max_words=3) == 1 / 3
    mean, std = grouped_distinct([["a b"], ["a a"]], 1)
    assert mean == 0.75 and std == 0.25


def test_reference_threshold_flags_about_five_percent():
    stories = [f"story number {i} went on a walk." for i in range(95)] + [LOOP.strip()] * 5
    ref = real_story_reference(stories, group_size=4, n_groups=5, seed=0)
    flagged = sum(repeated_ngram_rate(s) > ref["rep4_flag_threshold"] for s in stories)
    assert flagged == 5
    assert ref["proper_ending_rate"] == 1.0


def test_hand_labels_are_never_overwritten(tmp_path):
    path = str(tmp_path / "errors.csv")
    row = {"id": 0, "prompt": "p", "repetition": False, "incomplete": False, "hit_token_limit": False, "text": "t"}
    assert write_error_csv(path, [row]) == path
    with open(path, encoding="utf-8") as f:
        labelled = f.read().replace(",,,,t", "yes,,,,t")
    with open(path, "w", encoding="utf-8") as f:
        f.write(labelled)
    assert write_error_csv(path, [row]).endswith("errors_new.csv")
    with open(path, encoding="utf-8") as f:
        assert "yes" in f.read()


def test_peak_memory_is_at_least_current_memory():
    assert peak_memory_mb("cpu") >= memory_mb("cpu") * 0.99


def test_own_work_is_not_counted_as_other_programs():
    load = OtherCpuLoad()
    load.start()
    x = 0
    for i in range(3_000_000):  # keep this process busy for a moment
        x += i
    assert 0.0 <= load.stop() <= 100.0


def test_training_info_is_none_without_training_files(tmp_path):
    assert training_info(str(tmp_path)) is None
