import random
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from transformers import DataCollatorWithPadding
from termcolor import colored


class Ansi:
    green = '\033[32m'
    red = '\033[31m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'


def df_info(df: pd.DataFrame, isna_cols: list | None = None):
    display(df.head())
    print(f"{Ansi.bold}Shape:{Ansi.end} {df.shape}")
    print(f"\n{Ansi.bold}Missing values:{Ansi.end}")

    if isna_cols is not None:
        df = df[isna_cols]

    print(df.isna().sum().to_string(index=True))


def get_rare_labels_from_multihot(labels: np.ndarray) -> np.ndarray:
    """
    Функция принимает массив multi-hot векторов, рассчитывает количество раз, 
    которые встретился каждый класс и присваивает каждому объекту индекс самого 
    малочисленного из имеющихся классов.
    Полученный массив можно применять для стратификации данных, при разделении на выборки.
    """
    # Подсчитываем, сколько раз встречается каждый класс
    labels_cnt = np.sum(labels, axis=0)
    
    rare_labels = []
    for label in labels:
        # Применяем маску
        masked = labels_cnt[label == 1]
        # Находим индекс минимального значения в отфильтрованном массиве
        min_index_masked = np.argmin(masked)
        min_index = np.where(label == 1)[0][min_index_masked]
        rare_labels.append(min_index)

    return np.array(rare_labels)


#------------------
# Torch
#------------------
def torch_device() -> str:
    """Вовзращает строку 'cuda' при наличии оборудования, иначе 'cpu'."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"torch use: {colored(device, 'green', attrs=['bold'])}", 
          f"({torch.cuda.get_device_name()})"
          if torch.cuda.is_available() else "")
    return device


def get_model_size(model) -> str:
    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    model_size_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model_size: {Ansi.bold}{model_size:.2f}{Ansi.end}M params", end='')

    if model_size_train < model_size:
        print(f" || Trainable params: {Ansi.bold}{model_size_train:.2f}{Ansi.end}M "
              f"({Ansi.bold}{model_size_train / model_size:.4%}{Ansi.end})")


def get_requires_grad(model, detail=True, only_true=False) -> str:
    params = {name: p.requires_grad for name, p in model.named_parameters()}
    params = pd.Series(params)

    get_model_size(model)

    print(f"\n{Ansi.bold}Requires grad:{Ansi.end}")
    print(f"True:  {Ansi.bold}{params.sum()}{Ansi.end}")
    print(f"False: {Ansi.bold}{(~params).sum()}{Ansi.end}")

    if detail:
        print()
        if only_true:
            print(params[params].to_string(index=True))
        else:
            print(params.to_string(index=True))


#------------------
# transformers
#------------------
def check_unk_tokens(texts: list[str], tokenizer, mode='chars') -> list[str]:
    
    def get_unk_words(text) -> list[str]:
        tokenize = tokenizer(text, return_offsets_mapping=True, return_token_type_ids=False, 
                             return_attention_mask=False, return_tensors='pt')
        input_ids = tokenize['input_ids'][0]
        offsets_map = tokenize['offset_mapping'][0]

        idx_unk_token_in_text = np.where(input_ids == unk_token_id)[0]
        slices_unk_token = offsets_map[[idx_unk_token_in_text]]
        return [text[start:end] for start, end in slices_unk_token]

    unk_token_id = tokenizer.unk_token_id

    # Получаем массив уникальных слов, при токенизации которых возвращается специальный 
    # токен [UNK]. В данном массиве могут остаться известные слова к которым без пробела 
    # добавлен неизвестный символ, например "Ок👌" - если токенайзер не обучен на emoji.
    unk_words = set()
    for text in tqdm(texts):
        unk_words.update(get_unk_words(text))

    if mode =='words':
        return unk_words

    # Получившийся массив неизвестных слов склеиваем в одну строку а затем разбиваем 
    # на отдельные уникальные символы с помощью множества. Получившиеся символы так же 
    # прогоняем через токеназер и формируем финальный список уникальных неизвестных символов.
    unk_chars = set()
    for char in set("".join(unk_words)):
        unk_chars.update(get_unk_words(char))

    return sorted(list(unk_chars))


class DataCollatorWithPaddingPlusLabels(DataCollatorWithPadding):
    def __call__(self, features):
        # Извлечение и сохранение меток (labels)
        labels = [feature.pop('labels') for feature in features]
        # Применение базового коллатора для остальных данных
        batch = super().__call__(features)
        # Паддинг для labels с использованием -100
        batch['labels'] = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        return batch


class DataCollatorWithRandomMasking(DataCollatorWithPadding):
    def __init__(self, tokenizer, prob=.15, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.prob = prob

    def __call__(self, features):
        # Паддинг и подготовка данных с использованием базового collator
        batch = super().__call__(features)

        input_ids = batch['input_ids']

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in input_ids
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        probability_matrix = torch.zeros(input_ids.shape).masked_fill(~special_tokens_mask, value=self.prob)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 95% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.95)).bool() & masked_indices
        batch['input_ids'][indices_replaced] = self.tokenizer.mask_token_id

        # 5% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 1.)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        batch['input_ids'][indices_random] = random_words[indices_random]
    
        return batch