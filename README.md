# Análisis de Representaciones Generadas por Diferentes Modelos

En este repositorio se encuentra todo el código necesario para replicar los experimentos de mi tesis. El objetivo es analizar las representaciones intermedias generadas por diferentes modelos, considerando tanto aquellos que reciben como entrada texto como aquellos que reciben audio.

## Descripción

Para comparar cómo cambian las representaciones a lo largo de los modelos, se utilizará el subconjunto 'dev clean' de la base de datos LibriSpeech. Este subconjunto contiene 2703 audios con sus respectivas transcripciones. Se obtendrán las representaciones de cada una de las palabras de cada audio.

## Estructura de Carpetas

- **./datasets**: Contiene las siguientes carpetas:
  - **alignments**: Alineaciones de los audios y transcripciones.
  - **librispeech-clean-16k**: Audios en formato de 16kHz.
  - **librispeech-clean-24k**: Audios en formato de 24kHz.
  - Dentro de cada una de estas carpetas existe una subcarpeta 'dev-clean' con los datos correspondientes.

- **./experiments**: Aquí se guardan los embeddings generados para cada audio.

- **./layer-analysis**: Este es el directorio principal del repositorio desde donde se ejecutan todas las corridas de los experimentos.

## Cómo Ejecutar los Experimentos

1. **Preparar el entorno**: Instaladar todas las dependencias necesarias. Se hace ejecutando:
    ```bash
    pip install -r requirements.txt
    ```
2. **Descargar los datos**: Colocar los datos de LibriSpeech en la carpeta `./datasets/librispeech-clean-16k/dev-clean` y `./datasets/librispeech-clean-24k/dev-clean`.

3. **Ejecutar los experimentos**: Desde la carpeta `./layer-analysis`, ejecuta los scripts correspondientes para generar y analizar las representaciones.

## Procedimiento para replicar

Primero tenemos que obtener los alignments de cada uno de los audios, para esto debemos ejecutar el script `alignments_and_trans.py`, ubicado en la carpeta `alignments`. Este script acepta como argumentos el 'stride', 'sample_rate' y el 'path' donde se ubican los archivos con los alignments. El argumento 'sample_rate' puede ser 16 o 24, lo que indica la frecuencia de muestreo del audio. El 'stride' especifica el intervalo de tiempo entre ventanas de información capturadas.

Ejecutar:

```bash
python3 alignments/alignments_and_trans.py --sample_rate 16 --stride 10 --path '../datasets/alignments'
```

Se quiere obtener los embeddings de cada palabra, para lo cual necesitamos conocer los frames que los contienen. Por tanto, es fundamental conservar cada uno de los archivos JSON generados para los pasos siguientes y es muy importante generar un archivo según la configuración del modelo que estamos utilizando. 

### Obtención de Embeddings

Para procesar los embeddings de los audios, es necesario utilizar modelos específicos. Nuestro análisis se centra, en el caso de modelos de lenguaje, en Glove y BERT. Para usar otros modelos, se deberán adaptar el tokenizador y la inicialización del modelo. Si el modelo deseado está disponible en Hugging Face, la integración será sencilla. A continuación, se detallan los comandos para procesar los embeddings con diferentes modelos.

**Modelos de texto**

**Glove**

Para obtener los embeddings utilizando Glove, ejecutar el siguiente comando. Asegúrate de especificar correctamente el path al archivo de alignments.

```bash
python3 embeddings/glove_embedding.py --path 'alignments/audio_alignments_20_16.json'
```

**BERT**

Para obtener embeddings utilizando el modelo BERT, es necesario especificar el modelo, el path a los archivos de alignments y el dispositivo de procesamiento. En este caso, se puede utilizar cualquier archivo de alignments porque nos importa la transcripción entera y no la ubicación de cada palabra.

```bash
python3 embeddings/nlp_embedding.py --model bert-base-uncased --path 'alignments/audio_alignments_20_16.json' --device 'cuda'
```

**Modelos de Audio**

Consideramos modelos que fueron preentrenados para tareas de habla como para tareas mas generales de audio. En el caso de los modelos entrenados para ASR se utilizó wav2vec2, pero el código está diseñado para soportar cualquier modelo de procesamiento de habla implementado en s3prl. Es crucial enviar el archivo JSON que corresponde al alignment con la frecuencia de muestreo y el stride adecuados para cada modelo específico.

**Wav2vec2**

Para procesar los embeddings usando wav2vec2, se debe ejecutar el siguiente comando. Asegúrate de que el archivo JSON que se proporciona corresponda con la frecuencia de muestreo y el stride que el modelo requiere:

```bash
python3 embeddings/speech_embedding.py --model wav2vec2_large_960 --path 'alignments/audio_alignments_20_16.json' --device 'cuda'
```

También consideramos modelos que fueron entrenados para tareas generales de audio, actualmente el codigo soporta los modelos `encodecmae_base`, `BEATs_iter3` pero puede ser muy facilmente adaptado para calcular los embeddings de cualquier modelo implementado en https://github.com/mrpep/easyaudio. También hay que tener en cuenta la frecuencia de sampleo y el stride del modelo. Para obtener los embeddings hay que ejecutar:

```bash
python3 embeddings/audio_embeddings.py --model encodecmae_base --path 'alignments/audio_alignments_13.33_24.json' --device 'cuda'
```

Una vez que obtuvimos los embeddings para cada palabra de cada archivo podemos armarnos la "matriz" que contiene toda esta información. Eso se hace ejecutando:

```bash
python3 embeddings/matrix_embeddings.py --layer 5 --model 'wav2vec2' --words_path 'words_in_order1.json'
``` 

En donde tenemos que aclarar la capa que queremos representar asi como el `words_path` que indica las palabras comunes, es decir que tienen representación, a lo largo de todos los modelos.


### Experimentos

Consideramos diferentes experimentos:

**Centered Kernel Alignment (CKA):**

Calculamos Linear_CKA(X,Y) donde X e Y son las representaciones de cada palabra para un modelo y una capa en particular. El siguiente comando devuelve este valor para todas las combinaciones de capas de los dos modelos dados. 

```bash
python3 experiments/experiment_CKA.py --model1 'bert-base-uncased' --layer1 12 --model2 'wav2vec2_large_960' --layer2 24
``` 

## ASIF: