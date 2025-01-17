import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
sns.set_palette('muted')


#------------------
# Charts
#------------------
def hist_box_chart(data: list[float | int], title: str | None = None) -> None:
    """Построение гистограммы и boxplot для списка значений.

    Args:
        data: Список с числовыми значениями.
        title: Подпись графика.

    Returns:
        Функция отрисовывает графики. Значений не возвращает.
    """
    fig, ax = plt.subplots(2, figsize=(11, 3.5), sharex=True, gridspec_kw={"height_ratios": (1, .15)})
    
    sns.histplot(data=data, ax=ax[0], kde=True, bins=min(150, np.unique(data).shape[0]), stat='density')
    ax[0].axvline(x=np.median(data), color='black', alpha=.5, label=f'median ({np.median(data):.2f})')
    ax[0].axvline(x=np.mean(data), color='black', alpha=.5, label=f'mean ({np.mean(data):.2f})', linestyle='dashed')
    ax[0].set(title=title, xlabel='', ylabel='')
    ax[0].legend()

    sns.boxplot(data=data, ax=ax[1], boxprops={'alpha': .5}, orient='y')
    ax[1].set(xlabel='count', ylabel='')


def plot_counts(
    data: dict, 
    num_samples: int | None = None, 
    title: str | None = None, 
    ax=None,
    palette: str = 'dark:#69d_r',
) -> None:
    """Построение столбчатой диаграммы.

    Args:
        data: Словарь вида {название: количество значений}
        num_sumples: Общее количество объектов в выборке для вычисления процентного 
            соотношения. Необходимо, так как один объект может относится к нескольким
            классам.
        title: Подпись графика.
        ax: matplotlib axes. Если None, создается новая фигура и ось.

    Returns:
        Функция отрисовывает график. Значений не возвращает.
    
    """
    
    len_data = len(data)

    if num_samples is None:
        values = data.values()
        fmt_label = f"{{:.2f}}"
    else:
        values = [value / num_samples for value in data.values()]
        fmt_label = f"{{:.2%}}"
    

    figsize=(6, len_data * .19)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        # fig.set_size_inches(*figsize)
    
    sns.barplot(
        x=data.values(),
        y=data.keys(),
        hue=data.keys(),
        legend=False,
        palette=sns.color_palette(palette, len_data),
        ax=ax,
    )
    
    for container, label in zip(ax.containers, values):
        ax.bar_label(container, labels=[fmt_label.format(label)], fontsize=8, padding=1)

    for label in ax.get_yticklabels():
        label.set_fontfamily('monospace')
        
    ax.set_title(title)
    ax.set_xlabel('count')
    ax.set_ylabel('')

    if ax is None:
        plt.show()


def boxplot_cnt_words_per_labels(
    sentences: list, 
    label_names: list,
    labels_multihot: np.ndarray,
    title: str | None = None,
) -> None:
    sentences = np.array(sentences)
    cnt_words_each_labels = {}

    # Подсчет слов для объектов без класса
    indices = np.where(np.all(labels_multihot == 0, axis=1))[0]
    if len(indices) > 0:
        sents = sentences[indices]
        cnt_words, _ = cnt_words_chars(sents)
        cnt_words_each_labels['{EMPTY LABEL}'] = cnt_words
    
    # Для каждого класса составляем список с длинами текстов
    for idx, name in enumerate(label_names):
        indices = np.where(labels_multihot[:, idx] == 1)[0]
        sents = sentences[indices]
        cnt_words, _ = cnt_words_chars(sents)
        cnt_words_each_labels[name] = cnt_words

    # Для каждого списка вычисляем медиану и сортируем ключи в словаре 
    # в порядке возрастания медианной длины текста.
    median_cnt_words = {}
    for key, value in cnt_words_each_labels.items():
        median_cnt_words[key] = np.median(value)
        
    median_cnt_words = dict(sorted(median_cnt_words.items(), key=lambda x: x[1]))
    sorted_labels = list(median_cnt_words.keys())
    
    # Получаем отсортированный словарь
    cnt_words_each_labels = {name: cnt_words_each_labels[name] for name in sorted_labels}

    # Рисуем график
    plt.figure(figsize=(6, len(cnt_words_each_labels) * .2))
    sns.boxplot(cnt_words_each_labels, orient='y', boxprops={'alpha': .7})
    plt.yticks(fontfamily='monospace')
    plt.title(title, pad=15)
    plt.show()


#------------------
# Utils
#------------------
def cnt_words_chars(sentences: list[str]) -> tuple[list, list]:
    """Подсчет количества слов и символов в списке строк.

    Args:
        sentences: Список со строками.

    Returns:
        Список с количеством слов в каждой строке.
        Список с количеством символов в каждой строке.
    
    """
    cnt_words = [len(text.split()) for text in sentences]
    cnt_chars = [len(text) for text in sentences]
    return cnt_words, cnt_chars


def get_cnt_labels(
    label_names: list, 
    labels_multihot: np.ndarray, 
    sorted_values: bool = True
) -> tuple[dict, int]:
    """Подсчет количества элементов каждого класса в массиве onehot векторов.

    Args:
        label_names: Список с именами классов.
        labels_multihot: матрица multi-hot векторов. Каждая строка - объект.
            Каждый столбец - класс, индекс которого совпадает с индексом 
            в списке label_names.
        without_label: Название для группы объектов, не принадлежащих ни к одному классу.

    Returns:
        Отсортированный по ключам словарь вида:
        {название класса из label_names: количество единиц в столбце массива}
    
    """
    # Создаем словарь {название лейбла: количество объектов с данным лейблом}
    cnt = dict(zip(label_names, np.sum(labels_multihot, axis=0)))
    # Добавляем количество объектов, которые не принадлежат ни одному классу (при_наличии)
    cnt_empty_label = np.all(labels_multihot == 0, axis=1).sum()
    if cnt_empty_label > 0:
        cnt['{EMPTY LABEL}'] = cnt_empty_label

    if sorted_values:
        # Сортируем словарь по ключам
        cnt = dict(sorted(cnt.items(), key=lambda x: x[1], reverse=True))

    num_samples = labels_multihot.shape[0]
    return cnt, num_samples


def get_cnt_classes_per_object(labels_multihot: np.ndarray) -> tuple[dict, int]:
    id_cls, num_obj = np.unique(np.sum(labels_multihot, axis=1), return_counts=True)
    num_classes_per_object = {str(k): v for k, v in zip(id_cls, num_obj)}

    num_samples = labels_multihot.shape[0]
    return num_classes_per_object, num_samples


def get_cnt_assessment(data: pd.Series) -> tuple[dict, int]:
    cnt = data.value_counts(dropna=False).to_dict()
    cnt = dict(sorted(cnt.items(), reverse=True))
    cnt = {str(k if pd.isna(k) else int(k)): v for k, v in cnt.items()}

    num_samples = data.shape[0]
    return cnt, num_samples


def get_mean_assessments(
    label_names: list, 
    labels_multihot: np.ndarray,
    assessments: np.ndarray,
    sorted_values: bool = True
) -> dict:
    assessments = np.nan_to_num(assessments)
    # Сумма оценок по каждому классу
    weighted_assessments = labels_multihot.T @ assessments

    # Количество объектов каждого класса
    cnt = np.sum(labels_multihot, axis=0)
    mean_assessments = weighted_assessments / cnt

    mean_assessments = {name: ma for name, ma in zip(label_names, mean_assessments)}
    if sorted_values:
        mean_assessments = dict(sorted(mean_assessments.items(), key=lambda x: x[1], reverse=True))

    return mean_assessments