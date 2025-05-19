# Clasificador de SPAM con SVM

Este proyecto utiliza un modelo de Máquina de Vectores de Soporte (SVM) para clasificar mensajes de texto como SPAM o NO SPAM usando el dataset `spam.csv`.

## Requisitos

- Python 3.8+
- Las dependencias están en `requirements.txt`.

## Instalación

1. Clona este repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd SVM_SPAM_Classifier
   ```

2. Crea y activa un entorno virtual:

   ### Windows
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   ### macOS y Linux
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### Windows

1. Activa el entorno virtual (si aun no lo has hecho):
   ```bash
   .venv\Scripts\activate
   ```
2. Ejecuta el script:
   ```bash
   python main.py
   ```

### macOS & Linux

1. Activa el entorno virtual (si aun no lo has hecho):
   ```bash
   source .venv/bin/activate
   ```
2. Ejecuta el script:
   ```bash
   python main.py
   ```

## Alternativa: Uso sin entorno virtual

Si prefieres no usar un entorno virtual, puedes instalar las dependencias directamente en tu sistema:

```bash
pip install scikit-learn pandas
```

Luego, simplemente ejecuta el script:

```bash
python main.py
```

## Estructura del dataset

El archivo `spam.csv` debe tener las siguientes columnas:
- `v1`: Etiqueta (`spam` o `ham`)
- `v2`: Mensaje de texto

## Personaliza la clasificación

Puedes modificar o agregar emails de prueba en la lista `test_emails` dentro de `main.py` para probar el modelo con tus propios mensajes.

## Licencia

MIT 