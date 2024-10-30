import json

import pandas as pd
from sklearn.model_selection import train_test_split


class Classifier:

    def __init__(self, support_tickets = pd.DataFrame()):
        self.model = None
        self.labels = None
        self.support_tickets = None
        self.token_list = []

    def load_data(self, file_path: str = './assets/data.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            support_data = json.load(f)

        formatted_data = []
        for ticket in support_data['casos_suporte']:
            formatted_data.append({
                'token_assunto': ticket['token_assuntos'],  # Mantenha a lista de tokens
                'texto': ticket['texto']
            })

        self.support_tickets = pd.DataFrame(formatted_data)
        return self.support_tickets

    def prepare_data(self,
                     text_dataframe: pd.DataFrame = None,
                     label_dataframe: pd.DataFrame = None,
                     test_size: float = 0.2,
                     random_state: int = 42
                     ):
        text_dataframe = text_dataframe if text_dataframe is not None else self.support_tickets
        raw_x_train, raw_x_test, raw_train_labels, raw_test_labels = train_test_split(text_dataframe,
                                                                                      label_dataframe,
                                                                                      test_size=test_size,
                                                                                      random_state=random_state,
                                                                                      #stratify=label_dataframe
                                                                                      )
        return raw_x_train, raw_x_test, raw_train_labels, raw_test_labels

    def execute(self):
        self.support_tickets = self.load_data()

        self.support_tickets = self.load_data()

        raw_x_train, raw_x_test, raw_train_labels, raw_test_labels = self.prepare_data(
            text_dataframe=self.support_tickets['texto'].tolist(),
            label_dataframe=self.support_tickets['token_assunto']
        )

        labels_dict_train = {index: valor for index, valor in enumerate(raw_train_labels)}
        labels_dict_test = {index: valor for index, valor in enumerate(raw_test_labels)}

        print(f"Train Size: {len(labels_dict_train)}")
        print(f"Test Size : {len(labels_dict_test)}")