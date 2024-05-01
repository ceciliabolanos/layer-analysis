# Requirements
sudo apt-get update
sudo apt-get install sox libsox-fmt-all
pip install s3prl[all]
pip install torchaudio
pip install chardet
pip install PySoundFile librosa


# Procedimiento para obtener alignments de audio

Para obtener los alignments de cada uno de los audios, debemos ejecutar el script `alignments_and_trans.py`, ubicado en la carpeta `alignments`. Este script acepta como argumentos el 'stride', 'sample_rate' y el 'path' para almacenar el JSON generado. El argumento 'sample_rate' puede ser 16 o 24, lo que indica la frecuencia de muestreo del audio. El 'stride' especifica el intervalo de tiempo entre ventanas de información capturadas.

### Comando para ejecutar:

```bash
python3 alignments/alignments_and_trans.py --sample_rate 16 --stride 10 --path '../datasets/alignments'
```

Es crucial obtener los embeddings de cada palabra, para lo cual necesitamos conocer los frames que los contienen. Por tanto, es fundamental conservar cada uno de los archivos JSON generados para los pasos siguientes.


# Obtención de Embeddings

Para procesar los embeddings de los audios, es necesario utilizar modelos específicos. Nuestro análisis se centra, en el caso de modelos de lenguaje, en Glove y BERT. Para usar otros modelos, se deberán adaptar el tokenizador y la inicialización del modelo. Si el modelo deseado está disponible en Hugging Face, la integración será sencilla. A continuación, se detallan los comandos para procesar los embeddings con diferentes modelos.

## Modelos de Lenguaje

### Glove
Para obtener los embeddings utilizando Glove, ejecuta el siguiente comando. Asegúrate de especificar correctamente el path al archivo de alignments.

```bash
python3 embeddings/glove_embedding.py --path 'alignments/audio_alignments_20_16.json'
```
### BERT:

Para obtener embeddings utilizando el modelo BERT, es necesario especificar el modelo, el path a los archivos de alignments y el dispositivo de procesamiento. 

```bash
python3 embeddings/nlp_embedding.py --model bert-base-uncased --path 'alignments/audio_alignments_20_16.json' --device 'cuda'
```

## Modelos de Habla

Este repositorio utiliza modelos de wav2vec2, pero el código está diseñado para soportar cualquier modelo de procesamiento de habla implementado en s3prl. Es crucial enviar el archivo JSON que corresponde al alignment con la frecuencia de muestreo y el stride adecuados para cada modelo de habla específico.

### Wav2vec2

Para procesar los embeddings usando wav2vec2, que es el modelo principal usado en este repositorio, se debe ejecutar el siguiente comando. Asegúrate de que el archivo JSON que se proporciona corresponda con la frecuencia de muestreo y el stride que el modelo requiere:

```bash
python3 embeddings/speech_embedding.py --model wav2vec2_large_960 --path 'alignments/audio_alignments_20_16.json' --device 'cuda'
```

## Modelos de Audio

Actualmente el codigo soporta los modelos `encodecmae_base`, `BEATs_iter3` pero puede ser muy facilmente adaptado para calcular los embeddings de cualquier modelo implementado en https://github.com/mrpep/easyaudio. Como en los embeddings de habla, tener en cuenta la frecuencia de sampleo y el stride del modelo. Para obtener los embeddings hay que ejecutar:

```bash
    python3 embeddings/audio_embeddings.py --model encodecmae_base --path 'alignments/audio_alignments_13.33_24.json' --device 'cuda'
```


Una vez que obtuvimos los embeddings para cada palabra de cada archivo podemos armarnos la "matriz" que contiene toda esta información. Eso se hace ejecutando:

```bash
     python3 embeddings/matrix_embeddings.py --layer 5 --model 'wav2vec2' --words_path 'words_in_order1.json'
``` 

En donde tenemos que aclarar la capa que queremos representar asi como el `words_path` que indica las palabras comunes, es decir que tienen representación, a lo largo de todos los modelos.


------------------------------------------------------------------------

# Experiments instructions

Step by step instructions:

  Command: 
     ```
     python3 experiments/experiment_CKA.py --model1 'bert-base-uncased' --layer1 12 --model2 'wav2vec2_large_960' --layer2 24
     ```  
  This generates a matrix with the linear CKA score for each pair of layers.