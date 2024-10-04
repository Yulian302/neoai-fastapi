FROM public.ecr.aws/lambda/python:3.11-x86_64


WORKDIR /app

COPY ./models /app/models
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./main.py /app/