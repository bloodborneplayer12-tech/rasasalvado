from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# 🔹 Carga tu dataset desde Hugging Face Hub
dataset = load_dataset("Yeso888/mini-dataset-prueba")

# 🔹 Usa modelo pequeño
model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 🔹 Preprocesamiento
def preprocess(example):
    return tokenizer(example["pregunta"], truncation=True, padding="max_length", max_length=64)

dataset = dataset["train"].map(preprocess)

# 🔹 Crear el mapeo de etiquetas ANTES de dividir
unique_respuestas = list(set(dataset["respuesta"]))
label2id = {r: i for i, r in enumerate(unique_respuestas)}
id2label = {i: r for r, i in label2id.items()}

def encode_labels(example):
    example["label"] = label2id[example["respuesta"]]
    return example

dataset = dataset.map(encode_labels)

# 🔹 Ahora dividir en train/test
tokenized_dataset = dataset.train_test_split(test_size=0.2)

# 🔹 Cargar el modelo
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_respuestas),
    id2label=id2label,
    label2id=label2id
)

# 🔹 Configurar entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # <-- nombre actualizado
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

# 🔹 Entrenar
trainer.train()

# 🔹 Guardar modelo
model.save_pretrained("./modelo-violencia")
tokenizer.save_pretrained("./modelo-violencia")
print("✅ Entrenamiento completado y modelo guardado en ./modelo-violencia")
