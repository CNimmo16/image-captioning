FROM python:3.11-slim-bullseye

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY . /code

CMD ["/bin/sh", "/code/start.sh"]
