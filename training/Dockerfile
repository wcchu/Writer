FROM python:3.8

# install python packages
COPY training/requirements.txt /
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# install google cloud sdk
RUN curl --silent https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-322.0.0-linux-x86_64.tar.gz | tar -C /usr/local -xzf - && /usr/local/google-cloud-sdk/install.sh --usage-reporting=false --path-update=true --bash-completion=true --rc-path=/.bashrc

# copy only necessary files
COPY training/train.sh /
COPY training/learn.py /

CMD ./train.sh
