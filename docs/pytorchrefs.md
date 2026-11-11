# Referencia rápida de PyTorch

Enlaces oficiales y notas breves para las dudas más comunes al escribir
el código del TFG. No sustituye a la documentación, pero indica dónde mirar.

---

## 1. Dataset y DataLoader

**Tutorial oficial (empezar aquí):**
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

Qué explica:
- Qué es un Dataset y por qué necesita `__init__`, `__len__` y `__getitem__`.
- Ejemplo con CSV + carpeta de imágenes (muy parecido a nuestro caso).
- Qué hace DataLoader y para qué sirven `batch_size`, `shuffle`, `num_workers`.

**Referencia de la API:**
https://pytorch.org/docs/stable/data.html

Qué mirar:
- `torch.utils.data.Dataset`: clase base. Dos tipos: map-style (acceso por índice) e iterable-style.
- `torch.utils.data.DataLoader`: parámetros y comportamiento.
- Nota sobre `num_workers` y multiprocesamiento: en macOS/Windows hace falta
  `if __name__ == '__main__':` para evitar bucles.

**Regla mental:**
- `__init__`: se ejecuta una vez. Carga el CSV, guarda rutas y transform.
- `__len__`: devuelve el número de ejemplos.
- `__getitem__`: carga UNA imagen del disco y devuelve `(imagen, etiquetas)`.
  Las imágenes NO se cargan en `__init__`, solo rutas y etiquetas.

---

## 2. Modelos preentrenados (torchvision)

**Referencia:**
https://pytorch.org/vision/stable/models.html

Qué mirar:
- Cómo cargar pesos preentrenados: `models.resnet34(weights=models.ResNet34_Weights.DEFAULT)`.
- Qué significa "preentrenado en ImageNet".
- Cómo consultar las categorías de ImageNet: `weights.meta["categories"]`.
- Lista de arquitecturas disponibles (resnet, densenet, efficientnet, etc.).

**Nota para el TFG:**
- `model.fc` (ResNet) o `model.classifier` (DenseNet) es la última capa.
  Hay que sustituirla por `nn.Linear(in_features, num_classes)`.
- `in_features` se consulta con `model.fc.in_features` o `model.classifier.in_features`.

---

## 3. Transformaciones de imagen

**Referencia:**
https://pytorch.org/vision/stable/transforms.html

Qué mirar:
- `transforms.Compose`: encadena transformaciones.
- `Resize`, `RandomRotation`, `RandomHorizontalFlip`: data augmentation.
- `ToTensor`: PIL → tensor, escala a [0, 1].
- `Normalize`: normaliza con media/std. Para ImageNet:
  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].

---

## 4. Pérdida para multietiqueta

**Referencia:**
https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

Qué mirar:
- `BCEWithLogitsLoss`: combina sigmoid + binary cross-entropy.
- Es la pérdida estándar para clasificación multietiqueta.
- No aplicar sigmoid a la salida del modelo antes de esta pérdida
  (la aplica internamente).

---

## 5. Optimizadores

**Referencia:**
https://pytorch.org/docs/stable/optim.html

Qué mirar:
- `torch.optim.AdamW`: variante de Adam con weight decay desacoplado.
- `torch.optim.SGD`: alternativa clásica.
- Cómo pasar solo los parámetros entrenables:
  `filter(lambda p: p.requires_grad, model.parameters())`.

**Nota para el TFG:**
- En `feature extraction`, solo se entrenan los parámetros del clasificador.
- En `fine-tuning parcial`, se descongelan las últimas capas.
- En `fine-tuning completo`, todos los parámetros.

---

## 6. Dispositivos (CPU / CUDA / MPS)

**Referencia:**
https://pytorch.org/docs/stable/notes/mps.html

Qué mirar:
- `torch.device("mps")` para Apple Silicon.
- `torch.device("cuda")` para NVIDIA.
- `torch.device("cpu")` como fallback.
- `torch.backends.mps.is_available()` para comprobar disponibilidad.
- Limitaciones de MPS: no soporta float64, algunos operadores no están implementados.

**Nota para el TFG:**
- En Mac: `mps` si está disponible, si no `cpu`.
- En Colab: `cuda` si hay GPU, si no `cpu`.
- El código debe funcionar en ambos sin cambios (usar `device` como variable).

---

## 7. Guardar y cargar modelos

**Referencia:**
https://pytorch.org/tutorials/beginner/saving_loading_models.html

Qué mirar:
- `torch.save(model.state_dict(), "modelo.pth")`: guarda solo pesos.
- Para cargar: reconstruir el modelo y `model.load_state_dict(torch.load("modelo.pth"))`.
- Guardar el optimizador también si se quiere reanudar entrenamiento:
  `torch.save({"model": model.state_dict(), "optim": optimizer.state_dict()}, ...)`.

---

## 8. Métricas (no es PyTorch, es scikit-learn)

**Referencia:**
https://scikit-learn.org/stable/modules/model_evaluation.html

Qué mirar:
- `roc_auc_score`: AUROC.
- `average_precision_score`: AUPRC.
- `precision_score`, `recall_score`, `f1_score`: con `average="macro"` para multietiqueta.
- `zero_division=0` para evitar warnings cuando una clase no tiene positivos.

---

## 9. Errores comunes y dónde mirar

| Error | Causa probable | Dónde mirar |
|---|---|---|
| `RuntimeError: Expected all tensors to be on the same device` | Modelo y datos en dispositivos distintos | Sección 6 |
| `RuntimeError: DataLoader worker exited unexpectedly` | `num_workers > 0` sin `if __name__ == '__main__'` | Sección 1 |
| `RuntimeError: MPS backend out of memory` | Batch demasiado grande en Mac | Bajar `batch_size`, `torch.mps.empty_cache()` |
| `ValueError: Target size must be the same as input size` | Pérdida incorrecta para multietiqueta | Sección 4 |
| `TypeError: 'float' object is not subscriptable` | `loss` no es tensor (usar `loss.item()`) | Sección 7 |

---

## 10. Orden de lectura recomendado

1. Tutorial de Dataset y DataLoader (sección 1).
2. Modelos preentrenados (sección 2).
3. Pérdida multietiqueta (sección 4).
4. Optimizadores (sección 5).
5. Guardar/cargar (sección 7).
6. Métricas (sección 8).

Con eso cubres el 90% de lo que necesitas para el TFG.
