"""forward-chaining(時系列を守る)K分割交差検証。出典: 既存アプリ ml/cross_validation.py。

ランダム分割だと未来のデータで過去を予測してしまう(リーク)ため、常に「過去で学習→未来で検証」にする。
"""

from __future__ import annotations


def forward_chaining_splits(n_samples: int, *, n_splits: int = 5) -> list[tuple[range, range]]:
    if n_splits < 2:
        raise ValueError("n_splitsは2以上である必要があります")
    if n_samples < n_splits + 1:
        raise ValueError("n_samplesがn_splitsに対して少なすぎます")
    fold = n_samples // (n_splits + 1)
    return [
        (range(0, fold * i), range(fold * i, min(fold * (i + 1), n_samples)))
        for i in range(1, n_splits + 1)
    ]
