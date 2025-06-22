from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import json
import torch

# Cargar dataset
with open("training_pairs.json") as f:
    data = json.load(f)

    

print(f"[INFO] Total ejemplos cargados: {len(data)}")
print(f"[INFO] Primeros 3 ejemplos:\n{json.dumps(data[:3], indent=2)}")

# Convertir a InputExample
examples = [InputExample(texts=[item["query"], item["target"]], label=float(item["label"])) for item in data]

# Modelo base
word_embedding_model = models.Transformer('sentence-transformers/all-MiniLM-L6-v2')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# DataLoader
train_dataloader = DataLoader(examples, shuffle=True, batch_size=8)
train_loss = losses.CosineSimilarityLoss(model)

# Entrenamiento
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=10
)

# Guardar modelo
model.save('output/ia-homologador')
