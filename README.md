# Proyecto Ramo Programaciónn Cientifica

![unit: Departamento de Ingeniería de Sistemas y Computación](https://img.shields.io/badge/course-Departamento%20de%20Ingenier%C3%ADa%20de%20Sistemas%20y%20Computaci%C3%B3n-blue?logo=coursera)
![institution: Universidad Católica del Norte](https://img.shields.io/badge/institution-Universidad%20Cat%C3%B3lica%20del%20Norte-blue?logo=google-scholar)
![Last Commit](https://img.shields.io/github/last-commit/godiecl/template)

## Description

Este proyecto abarca dos aspectos, el primero es la limpieza y exploración de datos que se obtuvieron de mediciones de paginas del gobierno de Chile, los datos estan asociados a temperatura media, cantidad de mp10, cantidad de mp2.5 y cantidad de SO2.
Luego de la limpieza y exploración de los datos realizado en jupyter notebook guardamos los datos para luego utilizarlos para entrenar un modelo de regresion linea. Usando la herramienta pickle guardamos el modelo para re utilizarlo en la pagina web montada con la libreria streamlit.

Para iniciar el codigo de la web se debe usar el comando streamlit run main.py.

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
