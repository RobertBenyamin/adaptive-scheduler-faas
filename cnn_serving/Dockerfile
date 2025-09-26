FROM jovanvr97/cnn_serve_start

RUN pip install --upgrade pip
RUN pip install sockets pickle-mixin requests psutil js2py JPype1==1.5.1 redis numpy==1.23.5 jsonpickle boto3 Flask
RUN mkdir -p /app/pythonAction

ARG RUNNER_FILE=runner.py

ADD ${RUNNER_FILE} /app/pythonAction/
ADD app.py /app/pythonAction/
ADD dnld_blob.py /app/pythonAction/
ADD storage_helper.py /app/pythonAction/

EXPOSE 8081

# activate it for version 5
RUN pip install scikit-learn

# Keep the container running indefinitely
CMD ["python", "/app/pythonAction/runner.py"]
