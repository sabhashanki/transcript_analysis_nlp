FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "reg_login:app", "--host=0.0.0.0", "--port=8000", "--reload"]