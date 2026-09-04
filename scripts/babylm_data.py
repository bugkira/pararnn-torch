"""BabyLM Strict-Small download, 16k BPE, packed T=512 sequences."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from torch import Tensor

log = logging.getLogger("babylm_data")

EOS_TOKEN = "<|endoftext|>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [EOS_TOKEN, UNK_TOKEN, PAD_TOKEN]

TRAIN_SOURCES = (
    "BabyLM-community/BabyLM-2026-Strict-Small",
    "nilq/babylm-10M",
)


def _text_column(column_names: list[str]) -> str:
    for name in ("text", "content", "sentence", "data"):
        if name in column_names:
            return name
    raise KeyError(f"no text column in {column_names}")


def iter_hf_texts(dataset_id: str, *, config: str | None = None, split: str | None = None):
    """Yield document strings. Tries ``split`` then the first available split."""
    from datasets import load_dataset

    kw: dict = {}
    if config:
        kw["name"] = config
    if split:
        kw["split"] = split
    log.info("hf_load dataset=%s config=%s split=%s", dataset_id, config, split)
    ds = load_dataset(dataset_id, **kw)
    if hasattr(ds, "keys"):
        key = split if split in ds else next(iter(ds.keys()))
        ds = ds[key]
        log.info("hf_split_used=%s", key)
    col = _text_column(list(ds.column_names))
    for row in ds:
        text = row[col]
        if text and str(text).strip():
            yield str(text)


def load_train_texts() -> list[str]:
    last_err: Exception | None = None
    for source in TRAIN_SOURCES:
        try:
            texts = list(iter_hf_texts(source))
        except Exception as exc:
            last_err = exc
            log.warning("train_source_failed dataset=%s err=%s", source, exc)
            continue
        if texts:
            log.info("train_docs=%d dataset=%s", len(texts), source)
            return texts
    raise RuntimeError(f"could not load BabyLM train text ({last_err})")


def load_val_texts(spec: dict) -> list[str]:
    try:
        texts = list(
            iter_hf_texts(
                str(spec["dataset_val"]),
                config=str(spec.get("dataset_val_config") or "") or None,
                split=str(spec.get("dataset_val_split") or "dev"),
            )
        )
    except Exception as exc:
        log.warning("val_source_failed err=%s", exc)
        return []
    log.info("val_docs=%d", len(texts))
    return texts


def train_tokenizer(texts: list[str], vocab_size: int, dest: Path) -> Tokenizer:
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=int(vocab_size), special_tokens=list(SPECIAL_TOKENS))
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single=f"$A {EOS_TOKEN}",
        special_tokens=[(EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN))],
    )
    path = dest / "tokenizer.json"
    tokenizer.save(str(path))
    log.info("tokenizer_saved path=%s vocab=%d", path, tokenizer.get_vocab_size())
    return tokenizer


def load_tokenizer(dest: Path) -> Tokenizer:
    path = dest / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return Tokenizer.from_file(str(path))


def encode_texts(tokenizer: Tokenizer, texts: list[str], *, max_tokens: int | None) -> list[int]:
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    ids: list[int] = []
    for text in texts:
        piece = tokenizer.encode(text).ids
        ids.extend(piece)
        if not piece or piece[-1] != eos_id:
            ids.append(eos_id)
        if max_tokens is not None and len(ids) >= max_tokens:
            ids = ids[: int(max_tokens)]
            break
    return ids


def pack_ids(ids: list[int], seq_len: int) -> Tensor:
    t = int(seq_len)
    n = (len(ids) // t) * t
    if n < t:
        raise RuntimeError(f"need at least {t} tokens to pack, got {len(ids)}")
    return torch.tensor(ids[:n], dtype=torch.long).view(-1, t)


def cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "train_tokens.pt", cache_dir / "val_tokens.pt"


def prepare_packed(spec: dict, repo: Path) -> tuple[Tensor, Tensor, Tokenizer]:
    tok_dir = repo / spec["tokenizer_dir"]
    cache_dir = repo / spec["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = cache_paths(cache_dir)
    seq_len = int(spec["seq_len"])
    vocab = int(spec["vocab_size"])
    max_train = int(spec["max_train_tokens"])
    val_n = int(spec["val_report_sequences"])

    if train_path.exists() and val_path.exists() and (tok_dir / "tokenizer.json").exists():
        tokenizer = load_tokenizer(tok_dir)
        train = torch.load(train_path, map_location="cpu", weights_only=True)
        val = torch.load(val_path, map_location="cpu", weights_only=True)
        log.info(
            "packed_cache train_rows=%d val_rows=%d vocab=%d",
            int(train.shape[0]),
            int(val.shape[0]),
            tokenizer.get_vocab_size(),
        )
        return train, val, tokenizer

    train_texts = load_train_texts()
    if (tok_dir / "tokenizer.json").exists():
        tokenizer = load_tokenizer(tok_dir)
    else:
        tokenizer = train_tokenizer(train_texts, vocab, tok_dir)

    if train_path.exists():
        train = torch.load(train_path, map_location="cpu", weights_only=True)
        log.info("train_cache path=%s rows=%d", train_path, int(train.shape[0]))
    else:
        train_ids = encode_texts(tokenizer, train_texts, max_tokens=max_train)
        train = pack_ids(train_ids, seq_len)
        torch.save(train, train_path)
        log.info(
            "train_packed tokens=%d rows=%d seq_len=%d",
            int(train.numel()),
            int(train.shape[0]),
            seq_len,
        )

    if val_path.exists():
        val = torch.load(val_path, map_location="cpu", weights_only=True)
        log.info("val_cache path=%s rows=%d", val_path, int(val.shape[0]))
        return train, val, tokenizer

    val_texts = load_val_texts(spec)
    if not val_texts:
        hold = max(val_n, 32)
        if train.shape[0] <= hold:
            raise RuntimeError("train split too small to hold out a val set")
        val = train[-hold:].clone()
        train = train[:-hold]
        torch.save(train, train_path)
        log.warning("val_holdout_from_train rows=%d", int(val.shape[0]))
    else:
        val_ids = encode_texts(tokenizer, val_texts, max_tokens=val_n * seq_len)
        val = pack_ids(val_ids, seq_len)[:val_n]
        log.info("val_packed rows=%d", int(val.shape[0]))
    torch.save(val, val_path)
    return train, val, tokenizer
