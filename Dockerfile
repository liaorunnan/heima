FROM python:3.9 as requirements-stage

WORKDIR /tmp

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip config set global.trusted-host mirrors.aliyun.com

RUN pip install poetry
RUN pip install poetry-plugin-export

COPY ./pyproject.toml ./poetry.lock* /tmp/

RUN poetry lock

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

WORKDIR /code

COPY --from=requirements-stage /tmp/requirements.txt /code/requirements.txt

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip config set global.trusted-host mirrors.aliyun.com

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./rag  /code/rag

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8006"]