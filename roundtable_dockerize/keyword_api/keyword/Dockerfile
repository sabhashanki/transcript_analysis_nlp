FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en_core_web_md
RUN python3 -c 'import nltk; nltk.download("punkt")'
EXPOSE 8001
CMD ["uvicorn", "keyword_fastapi:app", "--host=0.0.0.0", "--port=8001", "--reload"]