# Simulador de Generadores de Números Aleatorios

Una aplicación web desarrollada con Streamlit para simular y analizar generadores de números aleatorios.

## Características

- Generación de números aleatorios con diferentes parámetros
- Análisis estadístico de los números generados
- Interfaz web interactiva
- Panel de configuración personalizable
- Visualización de resultados y análisis

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/Josecass01/simulador-de-generadores.git
cd simulador-de-generadores
```

2. Crea un entorno virtual:
```bash
python -m venv .venv
```

3. Activa el entorno virtual:
- En Windows:
```bash
.venv\Scripts\activate
```
- En Linux/Mac:
```bash
source .venv/bin/activate
```

4. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

Para ejecutar la aplicación:

```bash
streamlit run app.py
```

La aplicación se abrirá en tu navegador en `http://localhost:8501`

## Estructura del Proyecto

- `app.py` - Archivo principal de la aplicación Streamlit
- `config_panel.py` - Panel de configuración de parámetros
- `view.py` - Funciones de procesamiento de datos y visualización
- `requirements.txt` - Dependencias del proyecto

## Contribuir

Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir los cambios que te gustaría realizar.

## Licencia

Este proyecto está bajo la Licencia MIT.