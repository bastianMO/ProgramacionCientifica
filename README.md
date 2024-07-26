# Proyecto Ramo Programación Científica

![unit: Departamento de Ingeniería de Sistemas y Computación](https://img.shields.io/badge/course-Departamento%20de%20Ingenier%C3%ADa%20de%20Sistemas%20y%20Computaci%C3%B3n-blue?logo=coursera)
![institution: Universidad Católica del Norte](https://img.shields.io/badge/institution-Universidad%20Cat%C3%B3lica%20del%20Norte-blue?logo=google-scholar)

## Description

Este proyecto abarca dos aspectos: El primero es la limpieza y exploración de datos obtenidos de mediciones de páginas del gobierno de Chile. Los datos están asociados a la temperatura media, cantidad de MP10, cantidad de MP2.5 y cantidad de SO2. El segundo es la limpieza y exploración de los datos que se realizó en Jupyter Notebook, estos fueron guardados para usarlos posteriormente en el entrenamiento de un modelo de regresión lineal. Usando la herramienta Pickle, se guardó el modelo para reutilizarlo en la página web montada con la librería Streamlit.

Para iniciar el codigo de la web se debe usar el comando 
```sh 
streamlit run main.py.
```

Y si se quisiera reemplazar el modelo se deben hacer modificaciones a la función train_model()

## Tools

- **Language**: [Python 3.x](https://www.python.org/): Python is a programming language that lets you work quickly
  and integrate systems more effectively.
- **Libraries**:
      - black==24.4.2
      - coloredlogs==15.0.1
      - ipympl==0.9.4
      - matplotlib==3.8.4
      - notebook==7.2.1
      - pandas==2.2.2 # 2.9.0
      - plotly_express==0.4.1
      - python-dotenv==1.0.
      - streamlit==1.36.0
      - openpyxl==3.1.5
      - seaborn==0.13.2
      - scikit-learn==1.5.1
      - tensorflow==2.17.0

## Credits

- Bastian Muñoz Ordenes, [Universidad Católica del Norte](http://wwww.ucn.cl),
  Antofagasta, Chile.

## License

This project is open-sourced software licensed under the [Apache license](LICENSE).

![License](https://img.shields.io/github/license/godiecl/template)
