FROM python:3.10-slim

WORKDIR /app
COPY . /app/

RUN apt-get update && apt-get install -y gcc build-essential

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm

EXPOSE 5000
CMD ["python", "app.py"]
