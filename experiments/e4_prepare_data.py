"""E4 data prep: build the domain-shift token stream cache.

Domains (CN-network-accessible, reproducible sources):
  wiki -- WikiText (ModelScope 'modelscope/wikitext' 103-raw; fallback:
          wikitext-2-raw via PTB-style direct URL)
  news -- AG-News train.csv (classic GitHub mirror), title+description text
  code -- numpy v1.26.4 source tree (.py files), via GitHub codeload

Output: data/e4_stream_{domain}.pt with GPT-2 token id tensors.
"""

from __future__ import annotations

import csv
import io
import os
import tarfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

from data import _download_bytes, _download_text  # noqa: E402

AG_URL = ("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/"
          "data/ag_news_csv/train.csv")
NUMPY_URL = "https://codeload.github.com/numpy/numpy/tar.gz/refs/tags/v1.26.4"
PTB_WORD = ("https://raw.githubusercontent.com/wojzaremba/lstm/master/data/"
            "ptb.train.txt")


GPT2_CACHE = os.path.expanduser(
    "~/.cache/modelscope/AI-ModelScope/gpt2")


def get_tokenizer():
    # load straight from the modelscope cache dir — do NOT import modelscope
    # here (its native libs conflict with pyarrow at interpreter teardown)
    from transformers import GPT2TokenizerFast
    if os.path.isdir(GPT2_CACHE):
        return GPT2TokenizerFast.from_pretrained(GPT2_CACHE)
    from modelscope import snapshot_download
    return GPT2TokenizerFast.from_pretrained(
        snapshot_download("AI-ModelScope/gpt2"))


WT103_PARQUET = ("https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/"
                 "main/wikitext-103-raw-v1/train-00000-of-00002.parquet")


def text_wiki() -> str:
    """WikiText-103 text. The parquet is converted to plain txt by a
    SEPARATE pyarrow-only process (tools note: pyarrow read_table segfaults
    when torch is loaded in the same process on this Windows setup):

        python -c "import pyarrow.parquet as pq; ..."  -> data/wt103_text.txt
    """
    txt_path = os.path.join(DATA, "wt103_text.txt")
    if os.path.exists(txt_path):
        return open(txt_path, encoding="utf-8").read()
    parquet = os.path.join(DATA, "wt103_train0.parquet")
    if not os.path.exists(parquet):
        with open(parquet, "wb") as f:
            f.write(_download_bytes(WT103_PARQUET))
    import subprocess
    import sys
    code = (
        "import pyarrow.parquet as pq\n"
        f"tbl = pq.read_table(r'{parquet}', columns=['text'])\n"
        "parts, total = [], 0\n"
        "for t in tbl.column('text').to_pylist():\n"
        "    parts.append(t); total += len(t)\n"
        "    if total > 20_000_000: break\n"
        f"open(r'{txt_path}', 'w', encoding='utf-8').write(''.join(parts))\n")
    subprocess.run([sys.executable, "-c", code], check=True)
    return open(txt_path, encoding="utf-8").read()


def text_news() -> str:
    raw = _download_text(AG_URL)
    out = []
    for row in csv.reader(io.StringIO(raw)):
        if len(row) >= 3:
            out.append(row[1] + ". " + row[2])
    return "\n".join(out)


def text_code() -> str:
    blob = _download_bytes(NUMPY_URL)
    out = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            if m.name.endswith(".py") and m.size > 0:
                f = tar.extractfile(m)
                if f:
                    out.append(f.read().decode("utf-8", errors="replace"))
    return "\n\n".join(out)


def main():
    # phase 1: gather all raw texts FIRST (pyarrow must load before the
    # rust tokenizers lib — reverse order crashes natively on Windows)
    todo = {}
    for name, fn, cap in [("wiki", text_wiki, 4_000_000),
                          ("news", text_news, 3_000_000),
                          ("code", text_code, 3_000_000)]:
        cache = os.path.join(DATA, f"e4_stream_{name}.pt")
        if os.path.exists(cache):
            print(f"{name}: cached", flush=True)
            continue
        text = fn()
        print(f"{name}: {len(text):,} chars gathered", flush=True)
        todo[name] = (text, cap, cache)

    # phase 2: tokenize
    if todo:
        tok = get_tokenizer()
        for name, (text, cap, cache) in todo.items():
            ids = []
            chunk = 1_000_000
            for i in range(0, len(text), chunk):
                ids.extend(tok(text[i:i + chunk])["input_ids"])
                if len(ids) >= cap:
                    break
            ids = torch.tensor(ids[:cap], dtype=torch.long)
            torch.save(ids, cache)
            print(f"{name}: {len(ids):,} tokens -> {cache}", flush=True)


if __name__ == "__main__":
    main()
