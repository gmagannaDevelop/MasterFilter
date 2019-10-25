
# Cosas por hacer (internas) :
1. Crear función para el kernel Butterworth (**gus**) :white_check_mark:.

2. Función \_param\_check() (**gus**) : :white_check_mark:
   * Definir el cuerpo de \_param\_check() :white_check_mark: (**gus**)
   * Usarlo para reducir el boilerplate. :white_check_mark: (**gus**)

3. Debugguear kernel\_butterworth() (**gus**) :pencil:

4. Reimplementar filtros.
   Parecía que escribirlo en función del tipo de kernel era una buena idea, 
   pero en realidad lo mejor es crear dos funciones :
   * pasa\_bajos() que pueda computar las tres formulaciones.
   * pasa\_bandas() idem.

   Ya que pasa\_altos() es simplemente 1 - pasa\_bajos()
   Idem rechaza\_bandas() es  1 - pasa\_bandas()

5. Agregar protección para la divisón por cero. (**gus**)
   **np.finfo(float).eps**

# Puntos de la tarea : 
Fundamentos de procesamiento digital de imágenes : Funciones de filtrado en frecuencia

1. ( 2 puntos) Modificar las funciones vistas en clase que generan filtros de tipo ideal de la forma pasa-bajos, pasa-altos, pasa-bandas rechazo de bandas, generalizandolas para que se indique si el filtro es ideal, Gaussinao o de Butterworth

2. (3 puntos) Hacer una función que diseñe filtros notch ideales, de Gauss o de Butterworth.

3. (1 punto) Hacer una función maestra que diseñe filtros indicando: Tipo de filtro (pasa bajos, pasa altos, pasa bandas, rechazo de bandas, notch), con que forma del filtro (ideal, Gaussiano, Buttherworth) y los parámetros necesarios para realizar el diseño.

4. (1 punto) Una función que diseñe y aplique el filtro a una imagen.

5. Reporte los siguientes resultados:
    (1 punto) A la imagen que se llama mama.tiff
  
      Filtro pasa bajos ideal con wc=64,

      Filtro pasa bajos butt con wc=64, orden=2

      Filtro pasa bajos gauss con wc=64,

      Filtro pasa altos gauss con wc=64,

      Filtro pasa bandas gauss con wc1=54, wc2=74,

      Filtro rechazo de bandas gauss con wc1=54, wc2=74,

6. (2 puntos) A las figuras FigP0405(HeadCT_corrupted).tif y a RadiografiaRuidoCoherente.jpg intente quitar el ruido coherente que se observa, detalle el procedimiento que realizó y los resultados que obtuvo
