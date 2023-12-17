FROM python:3.8

RUN apt-get update && \
    apt-get install -y portaudio19-dev && \
    apt-get clean


CMD ["streamlit", "run", "streamlit_app.py"]
