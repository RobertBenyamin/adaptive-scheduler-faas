FROM jovanvr97/img_proc_start
#FROM fakhriakmal24/serverbase:1.0.0

RUN pip install --upgrade pip
RUN pip install sockets pickle-mixin requests psutil js2py JPype1==1.5.1 redis numpy jsonpickle boto3
RUN mkdir -p /app/pythonAction
ADD runner.py /app/pythonAction/
ADD app.py /app/pythonAction/
ADD dnld_blob.py /app/pythonAction/
ADD storage_helper.py /app/pythonAction/
# ADD faas-kubernetes-creds.json /app/

# ENV GOOGLE_APPLICATION_CREDENTIALS=/app/faas-kubernetes-creds.json

# Make port 8080 available to the world outside this container
# EXPOSE 8080 9999

# # Define environment variable for Flask
# ENV FLASK_APP=/app/pythonAction/runner.py.py
# ENV PYTHONUNBUFFERED=1

EXPOSE 8081

# activate it for version 5
# RUN pip install scikit-learn

# Keep the container running indefinitely
CMD ["python", "/app/pythonAction/runner.py"]
