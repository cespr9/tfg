# README

Info básica del trabajo 

-----


## Descripción:
Fine-tuning de modelos de visión para la detección de patologías en radiografías de tórax
  - Investigar como influye el preentrenamiento y la estrategia de fine-tuning en detectar distintas patologías torácicas.

## Título:
Evaluación de modelos de visión preentrenados y técnicas de fine-tuning para la detección automática de patologías en radiografías de tórax.

## Dataset:
NIH ChestX-ray14:
  -112.120 radiografías
  -30.805 pacientes
  -14 categorías de patologías
  -Etiquetas extraidas de informes radiológicos
  [Artículo base](https://pmc.ncbi.nlm.nih.gov/articles/PMC12428523/pdf/diagnostics-15-02215.pdf)

## Modelos a evaluar:

- ResNet50
- DenseNet21
- Modelo con preentrenamiento médico (si no piden autorización, o que no se haya entrenado con datos de mi test)

## Estrategias a evaluar:

- Feature extraction:
    - Mantener congeladas las capas del modelo preentrenado
    - Sustituir última capa por clasificador de 6 salidas
    - Entrenar solo esto
 
- Fine-tuning parcial:
    - Congelar 1ªs capas
    - Descongelar últimos bloques convolucionales
    - Entrenar estos bloques y el clasificador

- Fine-tuning completo:
    - Descongelar todas las capas
    - Ajustar pesos de toda la red
    - Tasa de aprendizaje reducida para el modelo preentrenado
    - (Aquí hay riesgo de overfitting)
 
## Métricas a evaluar:

- AUROC (área under curva ROC)
- AUPRC (área under curva PR)
- Precission
- Recall
- F1-score

