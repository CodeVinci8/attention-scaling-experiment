from __future__ import annotations

import copy
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
DATA_PATH = DATA_DIR / "tiny_shakespeare.txt"

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


@dataclass
class TrainConfig:
    batch_size: int = 32
    block_size: int = 64
    max_iters: int = 1000
    eval_interval: int = 100
    eval_iters: int = 20
    learning_rate: float = 3e-4

    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1

    seed: int = 42
    train_seed: int = 123
    eval_seed: int = 777
    grad_clip: float = 1.0


def get_device() -> str:
    """
    Возвращает cuda, если есть видеокарта, иначе cpu.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def download_tiny_shakespeare() -> None:
    """
    Скачивает датасет Tiny Shakespeare, если файла ещё нет.
    """
    DATA_DIR.mkdir(exist_ok=True)

    if DATA_PATH.exists():
        print(f"Dataset already exists: {DATA_PATH}")
        return

    print("Downloading Tiny Shakespeare dataset...")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, DATA_PATH)
    print(f"Dataset saved to: {DATA_PATH}")


def load_dataset():
    """
    Загружает текст, строит словарь символов и кодирует текст в числа.
    """
    download_tiny_shakespeare()

    text = DATA_PATH.read_text(encoding="utf-8")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str) -> list[int]:
        return [stoi[ch] for ch in s]

    data = torch.tensor(encode(text), dtype=torch.long)

    split_index = int(0.9 * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]

    return train_data, val_data, vocab_size, stoi, itos


def get_batch(
    data: torch.Tensor,
    cfg: TrainConfig,
    device: str,
    generator: torch.Generator,
):
    """
    Берёт случайный batch из текста.

    x — последовательность символов.
    y — та же последовательность, но сдвинутая на один символ вперёд.
    """
    max_start_index = len(data) - cfg.block_size - 1

    ix = torch.randint(
        low=0,
        high=max_start_index,
        size=(cfg.batch_size,),
        generator=generator,
    )

    x = torch.stack([data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix])

    return x.to(device), y.to(device)


class MultiHeadSelfAttention(nn.Module):
    """
    Многоголовое self-attention.

    Главное место эксперимента:

    if self.use_scaling:
        scores = scores / sqrt(head_size)

    Если use_scaling=False, деление отключается.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        use_scaling: bool,
    ):
        super().__init__()

        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.use_scaling = use_scaling

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_head, self.head_size)
        k = k.view(batch_size, seq_len, self.n_head, self.head_size)
        v = v.view(batch_size, seq_len, self.n_head, self.head_size)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = q @ k.transpose(-2, -1)

        if self.use_scaling:
            scores = scores / math.sqrt(self.head_size)

        scores = scores.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0,
            float("-inf"),
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)

        return self.proj(out)


class FeedForward(nn.Module):
    """
    Обычный feed-forward блок внутри Transformer.
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Один блок Transformer:
    LayerNorm -> SelfAttention -> residual
    LayerNorm -> FeedForward -> residual
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        use_scaling: bool,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
            use_scaling=use_scaling,
        )

        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd=n_embd, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyTransformerLanguageModel(nn.Module):
    """
    Маленькая character-level Transformer-модель.

    Задача:
    по последовательности символов предсказать следующий символ.
    """

    def __init__(
        self,
        vocab_size: int,
        cfg: TrainConfig,
        use_scaling: bool,
    ):
        super().__init__()

        self.block_size = cfg.block_size

        self.token_embedding = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embd=cfg.n_embd,
                    n_head=cfg.n_head,
                    block_size=cfg.block_size,
                    dropout=cfg.dropout,
                    use_scaling=use_scaling,
                )
                for _ in range(cfg.n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        batch_size, seq_len = idx.shape

        token_emb = self.token_embedding(idx)
        pos = torch.arange(seq_len, device=idx.device)
        pos_emb = self.position_embedding(pos)

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            batch_size, seq_len, vocab_size = logits.shape

            logits_flat = logits.view(batch_size * seq_len, vocab_size)
            targets_flat = targets.view(batch_size * seq_len)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss


@torch.no_grad()
def estimate_loss(
    model: TinyTransformerLanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    cfg: TrainConfig,
    device: str,
    seed: int,
) -> dict[str, float]:
    """
    Считает средний train loss и validation loss.
    """
    model.eval()

    result = {}

    for split_name, data in [("train", train_data), ("val", val_data)]:
        losses = []

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + (0 if split_name == "train" else 100_000))

        for _ in range(cfg.eval_iters):
            xb, yb = get_batch(data, cfg, device, generator)
            _, loss = model(xb, yb)

            if loss is None:
                raise RuntimeError("Loss is None during evaluation")

            losses.append(loss.item())

        result[split_name] = sum(losses) / len(losses)

    model.train()
    return result


def train_model(
    model: TinyTransformerLanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    cfg: TrainConfig,
    device: str,
    label: str,
) -> pd.DataFrame:
    """
    Обучает одну модель и возвращает историю loss.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    history = {
        "iter": [],
        "train_loss": [],
        "val_loss": [],
    }

    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(cfg.train_seed)

    torch.manual_seed(cfg.train_seed)

    progress = tqdm(range(cfg.max_iters), desc=f"Training {label}")

    for iter_num in progress:
        if iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters - 1:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                cfg=cfg,
                device=device,
                seed=cfg.eval_seed + iter_num,
            )

            history["iter"].append(iter_num)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])

            progress.set_postfix(
                train_loss=f"{losses['train']:.3f}",
                val_loss=f"{losses['val']:.3f}",
            )

        xb, yb = get_batch(train_data, cfg, device, train_generator)

        _, loss = model(xb, yb)

        if loss is None:
            raise RuntimeError("Loss is None during training")

        if torch.isnan(loss):
            print(f"Training stopped: loss became NaN for {label}")
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

    return pd.DataFrame(history)


def make_training_summary(
    history_scaled: pd.DataFrame,
    history_unscaled: pd.DataFrame,
) -> pd.DataFrame:
    """
    Создаёт таблицу 2 для статьи.
    """
    scaled_final = history_scaled.iloc[-1]
    unscaled_final = history_unscaled.iloc[-1]

    rows = [
        {
            "model": "Transformer со scaled attention",
            "scaling": "да",
            "final_train_loss": scaled_final["train_loss"],
            "final_validation_loss": scaled_final["val_loss"],
            "comment": "обучение завершено",
        },
        {
            "model": "Transformer без scaling",
            "scaling": "нет",
            "final_train_loss": unscaled_final["train_loss"],
            "final_validation_loss": unscaled_final["val_loss"],
            "comment": "обучение завершено",
        },
    ]

    return pd.DataFrame(rows)


def run_transformer_experiment():
    """
    Главная функция Part B.

    1. Загружает Tiny Shakespeare.
    2. Создаёт две модели с одинаковыми начальными весами.
    3. Обучает scaled и unscaled варианты.
    4. Сохраняет истории loss и таблицу результатов.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    cfg = TrainConfig()
    device = get_device()

    print(f"Using device: {device}")

    torch.manual_seed(cfg.seed)

    train_data, val_data, vocab_size, _, _ = load_dataset()

    train_data = train_data.to("cpu")
    val_data = val_data.to("cpu")

    print(f"Vocabulary size: {vocab_size}")
    print(f"Train tokens: {len(train_data)}")
    print(f"Validation tokens: {len(val_data)}")

    model_scaled = TinyTransformerLanguageModel(
        vocab_size=vocab_size,
        cfg=cfg,
        use_scaling=True,
    ).to(device)

    initial_state = copy.deepcopy(model_scaled.state_dict())

    model_unscaled = TinyTransformerLanguageModel(
        vocab_size=vocab_size,
        cfg=cfg,
        use_scaling=False,
    ).to(device)

    model_unscaled.load_state_dict(initial_state)

    print("\nTraining scaled Transformer...")
    history_scaled = train_model(
        model=model_scaled,
        train_data=train_data,
        val_data=val_data,
        cfg=cfg,
        device=device,
        label="scaled attention",
    )

    print("\nTraining unscaled Transformer...")
    history_unscaled = train_model(
        model=model_unscaled,
        train_data=train_data,
        val_data=val_data,
        cfg=cfg,
        device=device,
        label="unscaled attention",
    )

    history_scaled.to_csv(
        OUTPUT_DIR / "history_scaled_transformer.csv",
        index=False,
    )

    history_unscaled.to_csv(
        OUTPUT_DIR / "history_unscaled_transformer.csv",
        index=False,
    )

    table2 = make_training_summary(history_scaled, history_unscaled)

    table2.to_csv(
        OUTPUT_DIR / "table2_training_metrics.csv",
        index=False,
    )

    print("\nTable 2 results:")
    print(table2.round(4))

    return history_scaled, history_unscaled, table2