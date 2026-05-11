# Attention Scaling Experiment

Проект для статьи:

**«Математические основы механизма внимания в Transformer-моделях: от скалярного произведения до softmax»**

Цель проекта — экспериментально проверить, как деление `QKᵀ` на `√d_k` перед `softmax` влияет на распределение attention-весов.

## Идея

В Transformer используется scaled dot-product attention:

```text
Attention(Q, K, V) = softmax((QKᵀ) / √d_k) V
```

Гипотеза:

> Без деления на `√d_k` значения перед `softmax` становятся слишком большими, поэтому распределение внимания становится резким. Масштабирование делает attention-веса более стабильными.

## Структура проекта

```text
attention_experiment/
├── outputs/                       # Графики и таблицы результатов
├── main.py                        # Главный файл запуска
├── part_a_numpy_attention.py      # NumPy-эксперимент
├── part_b_tiny_transformer.py     # Эксперимент с малой Transformer-моделью
├── plotting.py                    # Построение графиков
├── utils.py                       # Вспомогательные функции
├── requirements.txt               # Зависимости
└── README.md
```

## Часть A. NumPy-эксперимент

Сравниваются два варианта attention:

```text
without scaling: softmax(QKᵀ)
with scaling:    softmax((QKᵀ) / √d_k)
```

Проверяются размерности:

```text
d_k = 8, 16, 32, 64, 128, 256
```

Метрики:

- средний максимальный attention-вес;
- энтропия распределения;
- heatmap attention-весов.

## Результаты NumPy-эксперимента

| d_k | max без scaling | entropy без scaling | max со scaling | entropy со scaling |
|---:|---:|---:|---:|---:|
| 8 | 0.624 | 1.055 | 0.314 | 1.934 |
| 16 | 0.725 | 0.745 | 0.316 | 1.929 |
| 32 | 0.803 | 0.519 | 0.317 | 1.926 |
| 64 | 0.870 | 0.338 | 0.323 | 1.919 |
| 128 | 0.903 | 0.247 | 0.315 | 1.933 |
| 256 | 0.934 | 0.166 | 0.319 | 1.926 |

Вывод:

- без scaling максимальный attention-вес растёт почти до `0.934`;
- со scaling максимальный вес остаётся около `0.31–0.32`;
- без scaling энтропия резко падает;
- со scaling энтропия остаётся высокой.

Это показывает, что деление на `√d_k` делает распределение внимания менее резким.

## Файлы результатов

После запуска в папке `outputs/` создаются:

```text
table1_attention_metrics.csv
fig2_heatmap_unscaled.png
fig3_heatmap_scaled.png
fig4_max_weight_vs_dk.png
fig4_extra_entropy_vs_dk.png
```

## Установка

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Установка зависимостей:

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python main.py
```

## Основной фрагмент логики

```text
scores = q @ k.T

if scaled:
    scores = scores / sqrt(d_k)

weights = softmax(scores)
```

Именно разница между `scaled=True` и `scaled=False` является центральной частью эксперимента.

## Часть B. Tiny Transformer

Вторая часть проекта предназначена для обучения двух малых character-level Transformer-моделей:

1. Transformer со scaled attention.
2. Transformer без scaling.

Метрики:

- training loss;
- validation loss;
- стабильность обучения.

Эта часть используется как дополнительная проверка влияния scaling на реальную модель.
