FROM jovanvr97/cnn_serve_start

RUN mkdir -p /app/pythonAction && chown -R appuser:appuser /app

RUN useradd -m appuser
USER appuser

WORKDIR /app/pythonAction

RUN pip install --upgrade pip
RUN pip install --user sockets pickle-mixin requests psutil js2py JPype1==1.5.1 redis numpy==1.23.5 jsonpickle boto3 Flask scikit-learn

ARG RUNNER_FILE=runner.py

ADD ${RUNNER_FILE} ./
ADD app.py ./
ADD dnld_blob.py ./
ADD storage_helper.py ./

EXPOSE 8081

CMD ["/home/appuser/.local/bin/python", "runner.py"]