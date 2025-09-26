FROM jovanvr97/ml_train_start
#FROM fakhriakmal24/serverbase:1.0.0

RUN pip install --upgrade pip
RUN pip install sockets pickle-mixin requests psutil js2py JPype1==1.5.1 redis Flask numpy boto3
RUN pip install scikit-learn torch==2.2.0 imgaug==0.4.0
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