import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

from classifier.Classifier import Classifier

from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, \
    RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModel
from classifier.CustomDataset import CustomDataset


def compute_metrics(p):
    predictions, labels = p

    # Usar um limiar para converter probabilidades em rótulos binários
    preds = (predictions > 0.5).astype(int)  # Ajuste o limiar conforme necessário

    # Calcular métricas de precisão, recall e F1 para multilabel
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='micro', zero_division=0  # Controla o comportamento com 0 previsões positivas
    )
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class BERT(Classifier):
    def __init__(self, support_tickets: pd.DataFrame = pd.DataFrame()):
        super().__init__(support_tickets)
        self.tokenizer = None
        self.mlb = None

    def tokenize(self,
                 train_texts: list,
                 test_texts: list,
                 pretrained_model_name: str = 'bert-base-uncased',
                 ):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

        # Tokenização
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')

        return train_encodings, test_encodings

    def prepare_train(self,
                      train_dataset: pd.DataFrame | CustomDataset,
                      test_dataset: pd.DataFrame | CustomDataset,
                      num_labels: int,
                      output_dir: str = './results',
                      num_train_epochs: int = 80,
                      per_device_train_batch_size: int = 10,
                      per_device_eval_batch_size: int = 10,
                      warmup_steps: int = 10,
                      weight_decay: float = 0.3,
                      logging_dir: str = './logs',
                      pretrained_model_name: str = 'bert-base-uncased'):
        # Carregando o modelo pré-treinado BERT
        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )

        # Configurando os argumentos de treinamento
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=logging_dir,
            logging_steps=500,  # Número de passos para registro
            eval_strategy="epoch",  # Avaliação ao final de cada época
            save_strategy="epoch",  # Salva o modelo apenas ao final de cada época
            #save_steps=500,  # Ajuste para salvar em intervalos maiores, dependendo do seu conjunto de dados
            save_total_limit=2,  # Mantém apenas os 2 últimos checkpoints para economizar espaço
            load_best_model_at_end=True,  # Carregar o melhor modelo ao final do treinamento
            metric_for_best_model="f1",  # Métrica para determinar o melhor modelo
            greater_is_better=True  # Se a métrica maior é melhor
        )

        # Criando o treinador
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics  # Método para calcular métricas
        )

        return trainer

    def execute(self):
        self.support_tickets = self.load_data()
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(self.support_tickets['token_assunto'])
        self.labels = self.mlb.transform(self.support_tickets['token_assunto'])

        raw_x_train, raw_x_test, raw_train_labels, raw_test_labels = self.prepare_data(
            text_dataframe=self.support_tickets['texto'].tolist(),
            label_dataframe=self.labels,
            test_size=0.2,
            random_state=42,
        )

        print(f"Train Size: {len(raw_train_labels)}")
        print(f"Test Size : {len(raw_test_labels)}")

        # Tokenizar os textos de treino e teste
        train_encodings, test_encodings = self.tokenize(train_texts=raw_x_train, test_texts=raw_x_test)

        # Criar datasets usando CustomDataset
        train_dataset = CustomDataset(encodings=train_encodings, labels=raw_train_labels)
        test_dataset = CustomDataset(encodings=test_encodings, labels=raw_test_labels)

        # Preparar e treinar o modelo
        trainer = self.prepare_train(train_dataset=train_dataset,
                                     test_dataset=test_dataset,
                                     num_labels=self.labels.shape[1],
                                     )
        trainer.train()

        # Fazer previsões
        predictions = trainer.predict(test_dataset)
        predicted_probs = torch.sigmoid(torch.tensor(predictions.predictions))  # Calcular probabilidades
        predicted_labels = (predicted_probs > 0.5).int().numpy()  # Converter para labels binárias

        # Relatório de classificação
        print("Predicted Labels:")
        print(predicted_labels)
        print("True Labels:")
        print(raw_test_labels)

        # Gerar o relatório de classificação
        report = classification_report(raw_test_labels,
                                       predicted_labels,
                                       zero_division=0,
                                       target_names=self.mlb.classes_)
        print(report)
