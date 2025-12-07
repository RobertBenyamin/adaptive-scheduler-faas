FROM jovanvr97/web_serve_start

WORKDIR /app/pythonAction

RUN pip install --upgrade pip
RUN pip install --no-cache-dir sockets pickle-mixin requests psutil js2py JPype1==1.5.1 redis Flask "numpy<1.24" jsonpickle boto3
# Use tensorflow-cpu to avoid CUDA/driver issues
RUN pip install tensorflow-cpu==2.12.0

# Set TensorFlow environment variables
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""

ARG RUNNER_FILE=runner.py

COPY ${RUNNER_FILE} ./runner.py
COPY app.py ./
COPY dnld_blob.py ./
COPY storage_helper.py ./

EXPOSE 8081
ENV PORT=8081

CMD ["python", "runner.py"]