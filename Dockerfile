FROM tensorflow/tensorflow:2.4.0
WORKDIR /app
COPY . .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt
CMD ./docker.sh
