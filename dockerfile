FROM pytorch/pytorch


RUN mkdir /code

COPY requirements.txt /code/

RUN pip install --upgrade pip
RUN pip install --upgrade pip setuptools wheel

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r /code/requirements.txt


COPY app.py /code/
COPY chatbot.py /code/
COPY santa.py /code/


COPY models /code/models/
COPY images /code/images
COPY templates /code/templates/

WORKDIR /code

EXPOSE 5000

CMD ["python", "app.py"]

