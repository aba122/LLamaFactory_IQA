import re
from typing import Tuple

# Strictly fetch the <answer> block and the first x.xx number inside it
PAT_BLOCK = re.compile(r"<answer>([\s\S]*?)</answer>")
PAT_NUM = re.compile(r"\b([1-5]\.\d{2})\b")


def find_ans_pos_via_prefix(full_text: str, tokenizer) -> Tuple[int, str]:
    """
    Locate the first token index of the numeric score inside <answer>…</answer>
    by regex + prefix re-tokenization. Works with slow or fast tokenizers.
    Returns (ans_pos:int, num_str:str).
    """
    match_block = PAT_BLOCK.search(full_text)
    if not match_block:
        raise ValueError("No <answer>...</answer> block found.")

    region = match_block.group(1)
    match_num = PAT_NUM.search(region)
    if not match_num:
        raise ValueError("No x.xx number in <answer>.")

    num_str = match_num.group(1)
    char_start = match_block.start(1) + match_num.start(1)

    prefix_ids = tokenizer(full_text[:char_start], add_special_tokens=False)["input_ids"]
    ans_pos = len(prefix_ids)
    return ans_pos, num_str
