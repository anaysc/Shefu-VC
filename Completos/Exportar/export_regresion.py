# ============================================================
# export_regresion.py
# Exporta el modelo de regresión de presentación de Shefu
# ============================================================

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ------------------------------------------------------------
# 1) Reconstruir el modelo exactamente como en entrenamiento
# ------------------------------------------------------------
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1  # para normalización de referencia
model = mobilenet_v3_small(weights=weights)

# Reemplazar la cabeza final (una sola salida continua en [0,1])
in_f = model.classifier[3].in_features
model.classifier[3] = nn.Sequential(
    nn.Linear(in_f, 1),
    nn.Sigmoid(),
    nn.Flatten(0)   # 🔹 Aplana la salida a escalar float (para TFLite)
)
model.eval()

# ------------------------------------------------------------
# 2) Cargar los pesos entrenados
# ------------------------------------------------------------
state = torch.load("checkpoints/best_model.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

# ------------------------------------------------------------
# 3) Exportar a formato ONNX
# ------------------------------------------------------------
dummy = torch.randn(1, 3, 224, 224)  # entrada simulada (NCHW)
torch.onnx.export(
    model,
    dummy,
    "shefu_regresion.onnx",
    input_names=["input"],
    output_names=["score"],  # 🔹 nombre coherente con flujo y app
    opset_version=13,
    dynamic_axes={"input": {0: "batch"}, "score": {0: "batch"}},
)
print("✅ Exportado ONNX: shefu_regresion.onnx")

# ------------------------------------------------------------
# 4) Referencia para normalización (usar en Flutter)
# ------------------------------------------------------------
# El modelo fue entrenado con normalización de ImageNet:
# mean = [0.485, 0.456, 0.406]
# std  = [0.229, 0.224, 0.225]
# 
# En Flutter (Dart), antes de pasar la imagen al modelo, normaliza cada pixel así:
#   normalized = (pixel / 255.0 - mean[i]) / std[i]
# Esto asegura que la inferencia produzca valores coherentes con el entrenamiento.
