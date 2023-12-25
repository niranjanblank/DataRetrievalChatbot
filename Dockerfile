FROM python:3.10
LABEL authors="noobieblank"

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit","run","app.py"]


