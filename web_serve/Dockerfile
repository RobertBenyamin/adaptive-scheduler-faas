FROM jovanvr97/web_serve_start

WORKDIR /app/pythonAction

RUN pip install --upgrade pip
RUN pip install --no-cache-dir sockets pickle-mixin requests psutil js2py JPype1==1.5.1 redis Flask "numpy<1.24" jsonpickle boto3 scikit-learn pytest
# RUN pip install tensorflow

ARG RUNNER_FILE=runner.py

COPY ${RUNNER_FILE} ./runner.py
COPY app.py ./
COPY dnld_blob.py ./
COPY storage_helper.py ./

EXPOSE 8081
ENV PORT=8081

CMD ["python", "runner.py"]