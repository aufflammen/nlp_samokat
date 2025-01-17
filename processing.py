import re
import random
import numpy as np
import pandas as pd
from googletrans import Translator, LANGUAGES
import asyncio
import nest_asyncio
nest_asyncio.apply() # Для совместимости с Jupyter Notebook


def get_multihot_tags(data: pd.Series, tag2id: dict) -> np.ndarray:
    """

    Args:

    Returns:
    
    """
    tags = np.zeros([data.shape[0], len(tag2id)], dtype=np.int32)

    for i, obj_tags in enumerate(data):
        if not pd.isna(obj_tags):
            obj_tags = obj_tags.strip('{}').split(',')
            # Проходим по каждому тегу для объекта и записываем multi-hot вектор
            for obj_tag in obj_tags:
                tags[i, tag2id[obj_tag]] = 1

    return tags
    

def drop_text_duplicates(data, col_labels):
    """ Удаление дубликатов текста и усреднение оценок и лейблов.
    
    Для одинаковых текстов результирующая пользовательская оценка берется 
        как среднее всех оценок для данного текста.

    Вычисление результирующего массива классов производится по следующему алгоритму:
        1. Если у объектов с выбранным текстом все классы нулевые, 
             то результирующий массив так же будет нулевой.
        2. Если только один объект будет иметь хотя бы один ненулевой класс, 
             то результирующим массив будет копией классов данного объекта.
        3. Если ненулевые классы будут иметь два объекта, то результирующий массив 
             будет состоять из классов, которые присутствуют в обоих ненулевых объектах.
        4. Если ненулевые классы будут у 3 и более объектов, то результирующий массив 
            будет состоять из классов, которые присутствуют хотя бы в половине ненулевых объектов.
    """

    new_data = pd.DataFrame()
    uniq_texts = data['text'].unique()
    
    for i, text in enumerate(uniq_texts):   
        slice_ = data[data['text'] == text]
        new_data.loc[i, 'assessment'] = slice_['assessment'].mean()
        new_data.loc[i, 'text'] = text
    
        labels_src = slice_[col_labels].to_numpy()
        cnt_nonzero = np.any(labels_src > 0, axis=1).sum()
        threshold = 1 if cnt_nonzero == 1 else max(2, (cnt_nonzero + 1) // 2)
        majority_vote = (np.sum(labels_src, axis=0) >= threshold).astype(int)
        new_data.loc[i, col_labels] = majority_vote
    
    new_data[col_labels] = new_data[col_labels].astype(int)
    return new_data


def prepare_data(
    df: pd.DataFrame, 
    col_labels: list | None = None,
) -> dict:
    """
    Args:

    Returns:
    
    """
    data = {}

    data['sentences'] = df['text'].to_numpy()
    data['urating'] = df['assessment'].to_numpy().astype(np.float32) / 6

    # Labels
    if col_labels is not None:
        data['labels'] = df[col_labels].to_numpy().astype(np.float32)

    return data


def x_y_split(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Представление датасета в виде двух numpy массивов, 
    для работы с классическими ML алгоритмами.
    """
    assessment = data['urating'].reshape(-1, 1)
    emb = data['emb']

    X = np.concatenate([assessment, emb], axis=1)
    y = data['labels']
    return X, y


def back_translation(
    texts, 
    source_lang: str = 'ru', 
    target_langs: list | None = None, 
    range_num_langs: tuple = (1, 2)
):
    """Перевод текста с использованием промежуточных случайных языков."""

    if target_langs is None:
        target_langs = list(LANGUAGES.keys())

    async def _back_translate_text(text, translator, intermediate_langs):
        """Асинхронный перевод текста через несколько языков."""
        current_text = text
        src_lang = source_lang
        for lang in intermediate_langs:
            translated = await translator.translate(text, src=src_lang, dest=lang)
            current_text = translated.text
            src_lang = lang

        back_translated = await translator.translate(current_text, src=src_lang, dest=source_lang)
        return back_translated.text

    async def main():   
        async with Translator() as translator:
            # Параллельная обработка
            tasks = [
                _back_translate_text(
                    text, 
                    translator, 
                    random.sample(target_langs, k=random.randint(*range_num_langs))
                )
                for text in texts
            ]
                
            text_translated = await asyncio.gather(*tasks)
            # Удаляем лишние пробелы
            return [re.sub(r'\s+', ' ', text) for text in text_translated]

    return asyncio.run(main())


def translation(text, src='ru', dest='en'):
    
    async def main():
        async with Translator() as translator:
            translated = await translator.translate(text, src=src, dest=dest)
            return translated.text

    return asyncio.run(main())