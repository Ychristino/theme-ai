import torch
from classifier.BERT import BERT

classifier = BERT()
classifier.execute()

text = 'Os eventos do esocial estão retornando com erro do governo por algum motivo que não tenho ideia.'
inputs = classifier.tokenizer(text, truncation=True, padding=True, return_tensors='pt')

with torch.no_grad():
    outputs = classifier.model(**inputs)

# A saída será os logits
logits = outputs.logits

# Aplicar a função sigmoid para obter as probabilidades (para multi-label)
probabilities = torch.sigmoid(logits)

# Obter as previsões com um limiar, por exemplo, 0.5
threshold = 0.5
predictions = (probabilities > threshold).int()

# Obter os nomes das labels diretamente do MultiLabelBinarizer
mlb = classifier.mlb  # As labels estão nas classes do MultiLabelBinarizer
labels = mlb.classes_  # Obtém os nomes das classes

# Armazenar os rótulos previstos
predicted_labels = []

# Iterar sobre as previsões e coletar rótulos encontrados
for i in range(predictions.shape[1]):  # iterar sobre o número de classes
    if predictions[0][i].item() == 1:  # se a previsão for 1 (presente)
        predicted_labels.append(labels[i])  # Adicionar o nome da label correspondente

# Imprimir os rótulos previstos
if predicted_labels:
    print(f"Text: '{text}'")
    print("Predicted labels:", ', '.join(predicted_labels))
else:
    print(f"Text: '{text}'")
    print("No labels predicted.")