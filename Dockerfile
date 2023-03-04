FROM python:3.8.10

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY ./myfirstnn.pickle /app/myfirstnn.pickle

COPY ./src /app/src


CMD ["python","./src/app.py"]
#docker push liyakhath/workshop:tagname