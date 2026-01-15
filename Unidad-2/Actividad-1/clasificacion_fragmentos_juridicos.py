# =========================================
# Actividad 1: Fine-tuning BERT (PyTorch directo, sin TrainingArguments)
# Modelo: dccuchile/bert-base-spanish-wwm-cased
# =========================================

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ---------
# 1) Dataset simulado (>= 200)
# ---------
LABELS = ["penal", "civil", "laboral", "familia"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

templates = {
    "penal": [
        "El acusado fue investigado por el delito de {delito} y se aportaron pruebas testimoniales.",
        "La Fiscalía imputó cargos por {delito} y solicitó medida de aseguramiento.",
        "Se discutió la tipicidad y antijuridicidad del hecho relacionado con {delito}."
    ],
    "civil": [
        "Se presentó demanda por incumplimiento contractual y se solicitó indemnización por perjuicios.",
        "El proceso trata sobre responsabilidad civil extracontractual por daños y perjuicios.",
        "Se pidió la nulidad del contrato y el restablecimiento de las prestaciones."
    ],
    "laboral": [
        "El trabajador alegó despido sin justa causa y reclamó el pago de prestaciones sociales.",
        "Se discutió el reconocimiento de horas extras, recargos y seguridad social.",
        "Se solicitó reintegro laboral por vulneración de estabilidad reforzada."
    ],
    "familia": [
        "Se solicitó fijación de cuota alimentaria y regulación de visitas del menor.",
        "El proceso aborda custodia y patria potestad con enfoque de interés superior del niño.",
        "Se tramitó divorcio y liquidación de sociedad conyugal."
    ]
}
delitos = ["hurto", "lesiones personales", "violencia intrafamiliar", "estafa", "homicidio"]

def make_sample(label: str) -> str:
    t = random.choice(templates[label])
    return t.format(delito=random.choice(delitos))

N_PER_CLASS = 60  # 240 ejemplos
texts, y = [], []
for lab in LABELS:
    for _ in range(N_PER_CLASS):
        texts.append(make_sample(lab))
        y.append(label2id[lab])

data = Dataset.from_dict({"text": texts, "label": y}).shuffle(seed=SEED)
splits = data.train_test_split(test_size=0.2, seed=SEED)
train_ds, test_ds = splits["train"], splits["test"]

print("Tamaño train:", len(train_ds), "| test:", len(test_ds))
print("Distribución (train):", Counter([id2label[i] for i in train_ds["label"]]))

# ---------
# 2) Tokenización
# ---------
model_name = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)

train_tok = train_ds.map(tokenize, batched=True)
test_tok = test_ds.map(tokenize, batched=True)

# Formato para PyTorch
train_tok = train_tok.remove_columns(["text"])
test_tok = test_tok.remove_columns(["text"])
train_tok.set_format("torch")
test_tok.set_format("torch")

collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(train_tok, batch_size=8, shuffle=True, collate_fn=collator)
test_loader = DataLoader(test_tok, batch_size=8, shuffle=False, collate_fn=collator)

# ---------
# 3) Modelo
# ---------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# ---------
# 4) Entrenamiento (fine-tuning)
# ---------
EPOCHS = 2
model.train()

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for batch in train_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(train_loader))
    print(f"Epoch {epoch}/{EPOCHS} - loss promedio: {avg_loss:.4f}")

# ---------
# 5) Evaluación (Accuracy + F1 macro)
# ---------
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        labels = batch["labels"].numpy().tolist()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = logits.argmax(dim=-1).cpu().numpy().tolist()

        all_labels.extend(labels)
        all_preds.extend(preds)

accuracy = accuracy_score(all_labels, all_preds)
f1_macro = f1_score(all_labels, all_preds, average="macro")

print("\nMétricas finales:")
print("Accuracy:", round(accuracy, 4))
print("F1 macro:", round(f1_macro, 4))

# ---------
# 6) Predicciones de ejemplo
# ---------
def predict(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = int(logits.argmax(dim=-1).cpu().item())
    return id2label[pred_id]

examples = [
    "Se solicitó reintegro laboral por estabilidad reforzada y pago de salarios caídos.",
    "La Fiscalía imputó cargos por hurto y solicitó medida de aseguramiento.",
    "Demanda por incumplimiento contractual y reclamación de perjuicios."
]

print("\nEjemplos de predicción:")
for e in examples:
    print("\nTexto:", e)
    print("Predicción:", predict(e))