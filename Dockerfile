FROM python:3.8

RUN apt-get update && \
    apt-get install -y portaudio19-dev && \
    apt-get clean

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

CMD ["streamlit", "run", "your_app.py"]
