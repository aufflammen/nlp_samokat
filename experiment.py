from collections.abc import Callable
from tqdm.auto import tqdm
from termcolor import colored
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay, 
    multilabel_confusion_matrix,
)
from transformers import DataCollatorWithPadding
from datasets import Dataset
from utils import Ansi
from custom_metric import binary_cross_entropy


def get_predict(model, dataloader, progress=True):
    device = next(model.parameters()).device
    results = []

    model.eval()
    with torch.no_grad():
        if progress:
            pbar = tqdm(dataloader, desc='predict', leave=True)
        else:
            pbar = dataloader

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            
            outputs = model(**batch)

            outputs = torch.sigmoid(outputs).cpu().numpy()
            results.append(outputs)

    return np.concatenate(results)


def generate_emb(
    model, 
    tokenizer, 
    sentences: list[str], 
    batch_size = 32, 
    num_epochs: int = 1,
    collate_fn: Callable | None = None, 
) -> list[list[float]]:
    """Генерация эмбеддингов моделью BERT.

    Args:
        model: Модель BERT.
        tokenizer: Токенизатор для модели.
        sentences: Список текстов.
        batch_size: Размер батча.

    Retruns:
        Список эмбеддингов для входных текстов.
    """
    # Batch dataloader
    if collate_fn is None:
        collate_fn = DataCollatorWithPadding(tokenizer)
        
    def tokenize_function(example):
        return tokenizer(example['sentences'], truncation=True, return_token_type_ids=False)
    
    dataset = Dataset.from_dict({'sentences': sentences})
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(['sentences'])
    dataset.set_format('torch')
    
    data_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)

    # Embeddings
    device = model.device.type
    model.eval()
    with torch.no_grad():
        embeddings = []
        pbar = tqdm(range(num_epochs * len(data_loader)), desc='generate_embeddings')
        for _ in range(num_epochs):
            for batch in data_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embedding.cpu().numpy())
                pbar.update(1)

    return np.concatenate(embeddings)


def color_value(value, params, attrs: list | None = None) -> str:
    fmt = f"{{:{params}}}"
    
    if value > 0.7:
        color = 'light_green'
    elif value > 0.3:
        color = 'light_yellow'
    else:
        color = 'light_red'

    return colored(fmt.format(value), color, attrs=attrs)


def get_metrics(pred_proba: np.ndarray, target: np.ndarray, threshold: float = 0.5, detail_stat=True) -> str:
    """Расчет нескольких метрик и вывод их в функции print."""
    pred = (pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(target, pred)
    bce = binary_cross_entropy(pred_proba, target).mean()

    metrics = {
        'f1': f"{'F1:':<10}",
        'precision': f"{'Precision:':<10}",
        'recall': f"{'Recall:':<10}",
    }
    average_options = ('micro', 'macro', 'weighted', 'samples')
    for average in average_options:
        avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
            target, pred, average=average, zero_division=0
        )
        metrics['f1'] += f"{Ansi.bold}{avg_f1:>10.2f}{Ansi.end}"
        metrics['precision'] += f"{avg_p:>10.2f}"
        metrics['recall'] += f"{avg_r:>10.2f}"

    
    print(f'Accuracy: {Ansi.bold}{accuracy:>12.4f}{Ansi.end} (samples)')
    print(f'BCE:      {Ansi.bold}{bce:>12.4f}{Ansi.end}\n')

    # f1, presicion, recall
    print(f"{'':>11}" +
          "".join(f"{opt:>10}" for opt in average_options))
    for metric in metrics.values():
        print(metric)
    print()

    # Common confusion matrix
    _, ax = plt.subplots(figsize=(2, 2))
    ConfusionMatrixDisplay.from_predictions(target.flatten(), 
                                            pred.flatten(), 
                                            ax=ax, 
                                            cmap='Blues', 
                                            labels=[0, 1],
                                            values_format='d',
                                            colorbar=False)

    ax.set_title('common confusion matrix', fontsize='small')
    ax.grid(False)
    plt.show()

    # metrics for each label
    if detail_stat:
        p, r, f1, s = precision_recall_fscore_support(target, pred, average=None, zero_division=0)
        # Заголовок
        print(f'\nmetrics for each class:')
        print(f"{Ansi.bold}{'id':>2} {'F1':>10} {'P':>6} {'R':>6} {'num':>7}{Ansi.end}")
        # Значения для каждого класса
        for i in range(p.shape[0]):
            f1_color = color_value(f1[i], '>10.0%', ['bold'])
            p_color = color_value(p[i], '>6.0%')
            r_color = color_value(r[i], '>6.0%')
            print(f"{i:>2} {f1_color} {p_color} {r_color} {s[i]:>7}")
            

def show_predict(
    pred: np.ndarray, 
    target: np.ndarray, 
    sentences: list, 
    id2label: dict, 
    nums: int = 3, 
    random_samples: bool = False
) -> None:

    def color_proba_black_gray(idx: int, label: int) -> str:
        value = pred[idx, label]
        color = 'black' if value >= .5 else 'light_grey'
        percent = f"({value:.1%})"
        return colored(f"{percent:>7}", color)

    def color_label_green_red(label: int, true_labels: np.array) -> str:
        color = 'green' if label in true_labels else 'red'
        return colored(id2label[label], color)

    bce = binary_cross_entropy(pred, target)

    if random_samples:
        indices = np.random.randint(target.shape[0], size=nums)
    else:
        indices = np.argsort(bce)[::-1][:nums]
    
    for idx in indices:
        print(f"{Ansi.bold}Sentence:{Ansi.end} {sentences[idx]}")
        print(f"{Ansi.bold}BCE:{Ansi.end} {bce[idx]:.4f}")
    
        print(f"\n{Ansi.bold}True labels:{Ansi.end}")
        true_labels = np.where(target[idx])[0]
        for label in true_labels:
            print(f"{label:>4}: {color_proba_black_gray(idx, label)} {id2label[label]}")
    
        pred_labels = np.argsort(pred[idx])[::-1][:5]
        print(f"\n{Ansi.bold}Pred labels (top 5):{Ansi.end}")
        for label in pred_labels:
            label_color = color_label_green_red(label, true_labels)
            print(f"{label:>4}: {color_proba_black_gray(idx, label)} {label_color}")

        if nums > 1:
            print('=' * 90)