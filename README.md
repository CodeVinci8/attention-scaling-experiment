# Attention Scaling Experiment

Проект для научной статьи:

**«Математические основы механизма внимания в Transformer-моделях: от скалярного произведения до softmax»**

Проект проверяет, как деление `QKᵀ` на `√d_k` перед функцией `softmax` влияет на распределение attention-весов в Transformer-моделях.

## Кратко

В Transformer используется формула scaled dot-product attention:

```text
Attention(Q, K, V) = softmax((QKᵀ) / √d_k) V
```

Главная идея эксперимента:

> Если не делить `QKᵀ` на `√d_k`, значения перед `softmax` становятся слишком большими. Из-за этого attention может стать слишком резким: почти весь вес внимания уходит одному токену. Масштабирование делает распределение внимания более равномерным.

## Что такое attention

Attention — это механизм, который помогает модели понять, какие элементы входной последовательности важнее для текущего элемента.

Например, если модель обрабатывает текст, attention помогает ей определить, на какие слова или символы нужно сильнее “смотреть”, чтобы предсказать следующий токен.

Упрощённо:

```text
Q — query, что текущий токен ищет
K — key, признаки других токенов
V — value, информация, которую можно взять
QKᵀ — оценка похожести токенов
softmax — превращение оценок в веса внимания
```

## Структура проекта

```text
attention_experiment/
├── data/                          # Датасет Tiny Shakespeare
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

В первой части проекта я не обучал модель, а напрямую проверил математику attention на случайных матрицах `Q` и `K`.

Сравнивались два варианта:

```text
without scaling: softmax(QKᵀ)
with scaling:    softmax((QKᵀ) / √d_k)
```

Проверялись размерности:

```text
d_k = 8, 16, 32, 64, 128, 256
```

Метрики:

- средний максимальный attention-вес;
- энтропия распределения;
- heatmap attention-весов.

### Результаты NumPy-эксперимента

| d_k | max без scaling | entropy без scaling | max со scaling | entropy со scaling |
|---:|---:|---:|---:|---:|
| 8 | 0.624 | 1.055 | 0.314 | 1.934 |
| 16 | 0.725 | 0.745 | 0.316 | 1.929 |
| 32 | 0.803 | 0.519 | 0.317 | 1.926 |
| 64 | 0.870 | 0.338 | 0.323 | 1.919 |
| 128 | 0.903 | 0.247 | 0.315 | 1.933 |
| 256 | 0.934 | 0.166 | 0.319 | 1.926 |

Вывод:

- без scaling максимальный attention-вес вырос почти до `0.934`;
- со scaling максимальный attention-вес остался около `0.31–0.32`;
- без scaling энтропия резко снизилась;
- со scaling энтропия осталась высокой и стабильной.

Это показывает, что деление на `√d_k` делает распределение внимания менее резким.

## Часть B. Tiny Transformer

Во второй части проекта была обучена маленькая character-level Transformer-модель.

Важно:

> Я обучил маленькую учебную Transformer-модель, которая предсказывает следующий символ в тексте.

Датасет:

```text
Tiny Shakespeare
```

Задача модели:

```text
по предыдущим символам предсказать следующий символ
```

Сравнивались две версии одной и той же архитектуры:

1. Transformer со scaled attention.
2. Transformer без scaling.

### Результаты обучения

| Модель | Scaling | Train loss | Validation loss | Комментарий |
|---|---|---:|---:|---|
| Transformer со scaled attention | да | 2.413 | 2.409 | обучение стабильное |
| Transformer без scaling | нет | 2.387 | 2.383 | обучение стабильное, loss немного ниже |

Вывод по обучению:

Обе модели обучались стабильно. В данном запуске модель без scaling показала немного меньший loss, но разница была небольшой. Поэтому основной вывод проекта делается не по loss маленькой модели, а по прямому NumPy-анализу attention-весов.


## Файлы результатов

После запуска проекта в папке `outputs/` создаются:

```text
table1_attention_metrics.csv
fig2_heatmap_unscaled.png
fig3_heatmap_scaled.png
fig4_max_weight_vs_dk.png
fig4_extra_entropy_vs_dk.png
history_scaled_transformer.csv
history_unscaled_transformer.csv
table2_training_metrics.csv
fig5_train_loss.png
fig6_val_loss.png
```

Назначение файлов:

- `table1_attention_metrics.csv` — таблица метрик NumPy-эксперимента;
- `fig2_heatmap_unscaled.png` — heatmap attention без scaling;
- `fig3_heatmap_scaled.png` — heatmap attention со scaling;
- `fig4_max_weight_vs_dk.png` — влияние `d_k` на максимальный attention-вес;
- `fig4_extra_entropy_vs_dk.png` — влияние `d_k` на энтропию;
- `history_scaled_transformer.csv` — история обучения scaled Transformer;
- `history_unscaled_transformer.csv` — история обучения unscaled Transformer;
- `table2_training_metrics.csv` — итоговые метрики обучения;
- `fig5_train_loss.png` — график training loss;
- `fig6_val_loss.png` — график validation loss.

## Установка

Создать виртуальное окружение:

```bash
python -m venv .venv
```

Активировать окружение на Windows:

```bash
.venv\Scripts\activate
```

Активировать окружение на Linux/macOS:

```bash
source .venv/bin/activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

## Запуск

Запустить только NumPy-эксперимент:

```bash
python main.py --part a
```

Запустить только Tiny Transformer:

```bash
python main.py --part b
```

Запустить всё:

```bash
python main.py --part all
```

## Главная логика эксперимента

Основная разница между двумя вариантами attention находится здесь:

```text
scores = q @ k.T

if scaled:
    scores = scores / sqrt(d_k)

weights = softmax(scores)
```

Если `scaled=True`, используется стандартный scaled dot-product attention.

Если `scaled=False`, деление на `√d_k` отключается.

## Итоговый вывод

NumPy-эксперимент показал, что scaling сильно влияет на сами attention-веса: без деления на `√d_k` распределение становится резким, а при scaling остаётся более ровным.

Эксперимент с маленьким Transformer показал, что на малой модели обе версии способны обучаться стабильно. Поэтому главный результат проекта состоит в том, что scaling стабилизирует softmax-распределение attention-весов, а его влияние на итоговый loss зависит от размера модели, параметров обучения и архитектурных стабилизаторов.
