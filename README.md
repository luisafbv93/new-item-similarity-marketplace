# new-item-similarity-marketplace

## Take Home: New Items

### Luisa Benavides

En este ejercicio se realiza un modelo de predicción para determinar los 5 ítems ya existentes más parecidos a un ítem nuevo que se ingrese. 
El dataset original cuenta con 100.000 registros de items extraidos del marketplace en MercadoLibre, caracterizados a través de 26 diferentes columnas. 

Para asegurar su funcionamiento se incluye:
- Jupyter notebook con el desarrollo del análisis completo y los resultados obtenidos.
 **Cuaderno_Solucion_New_Items_LuisaB.ipynb**
- Modelo de k-modes entrenado para agrupar los ítems de acuerdo a algunas de sus variables representativas. 
**model_kmodes.joblib**
- Tensor de características para los títulos de los ítems, construido previamente para facilitar la ejección.
**tensor_embeddingsAll.pt**
- Archivo de python con la solución final.
 **Solucion_New_Items_LuisaB.py**
- Modelo preentrenado usado para determinar similitud entre frases. La documentación de este modelo se puede encontrar en el link [sentence-similarity].
- requirements.txt

#### Ejecución:
- Inicialmente se debe realizar la instalación de los paquetes necesarios usando el archivo requirements.txt
- La ejecución se realiza desde el archivo de python **Solucion_New_Items_LuisaB.py** que toma como insumos dos archivos .cvs, uno debe contener todos los ítems existentes y en el otro se incluye solo el ítem nuevo que se va a revisar. 
- El resutlado de esta ejecución es un archivo .cvs con los títulos y códigos de cada uno de los 5 elementos que más se asemejan al de interes de acuerdo con los criterios definidos por el modelo. 


[sentence-similarity]: <https://huggingface.co/hiiamsid/sentence_similarity_spanish_est>
