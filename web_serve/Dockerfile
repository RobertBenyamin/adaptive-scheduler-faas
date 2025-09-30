FROM jovanvr97/web_serve_start

RUN useradd -m appuser

USER appuser

WORKDIR /app/pythonAction

RUN pip install --upgrade pip
RUN pip install --user sockets pickle-mixin requests psutil js2py JPype1==1.5.1 redis Flask numpy jsonpickle boto3
RUN pip install --user scikit-learn

ARG RUNNER_FILE=runner.py

ADD ${RUNNER_FILE} ./
ADD app.py ./
ADD dnld_blob.py ./
ADD storage_helper.py ./

EXPOSE 8081

CMD ["/home/appuser/.local/bin/python", "runner.py"]