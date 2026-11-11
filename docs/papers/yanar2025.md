# Paper Yanar 2025

- Dataset: NIH ChestX-ray14, 112.120 imágenes, 14 patologías, split 70/10/20 sin solapamiento de pacientes.
- Comparan 14 modelos en 3 familias: CNN, Transformers, Mamba
- Preprocesado: Reescalado a 224x224, flip horizontal 50%, normalización con mean/std del dataset
- Métrica: AUROC por clase y media
- Mejores resultados: ConvFormer, EfficientNet, CaFormer
- Coste computacional: Tabla 4 -> El más eficiente es EfficientNet
- Limitaciones: Etiquetas débiles, desbalanceo
- Entrenan todo el modelo bajo un protocolo común, pero no evalúan feature extraction vs fine-tuning parcial vs fine-tuning completo
