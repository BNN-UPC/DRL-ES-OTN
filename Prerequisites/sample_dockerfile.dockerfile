# You can change the base image to use.
# Note that this code is meant to work with Ubuntu 20.04 and the amd64 arquitecture, changes may be necessary
FROM amd64/ubuntu:20.04

RUN apt-get update
RUN apt-get install software-properties-common -qq

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install python3.6 python3.6-dev python3-dev python3.6-venv python3-virtualenv openmpi-bin openmpi-doc libopenmpi-dev git tmux clang g++ build-essential -qq

RUN mkdir -p /code
RUN chmod a+rwx /code


RUN ln -sf /bin/bash /bin/sh
RUN useradd -ms /bin/bash  ubuntuUser
USER ubuntuUser:100


WORKDIR /code
RUN virtualenv -p python3.6 ES_ENV
RUN /code/ES_ENV/bin/pip install pip

RUN /code/ES_ENV/bin/pip install absl-py==0.13.0 astunparse==1.6.3 cachetools==4.2.2 certifi==2021.5.30 charset-normalizer==2.0.2 cloudpickle==1.6.0 cycler==0.10.0 dataclasses==0.8 decorator==4.4.2 flatbuffers==1.12 gast==0.3.3 google-auth==1.33.0 google-auth-oauthlib==0.4.4 google-pasta==0.2.0 grpcio==1.32.0 gym==0.18.3 h5py==2.10.0 idna==3.2 importlib-metadata==4.6.1 Keras==2.4.3 Keras-Preprocessing==1.1.2 kiwisolver==1.3.1 Markdown==3.3.4 matplotlib==3.3.4 mpi4py==3.0.3 networkx==2.5.1 numpy==1.19.5 oauthlib==3.1.1 opt-einsum==3.3.0 pandas==1.1.5 Pillow==8.2.0 pkg_resources==0.0.0 protobuf==3.17.3 pyasn1==0.4.8 pyasn1-modules==0.2.8 pyglet==1.5.15 pyparsing==2.4.7 python-dateutil==2.8.2 pytz==2021.1 PyYAML==5.4.1 requests==2.26.0 requests-oauthlib==1.3.0 rsa==4.7.2 scipy==1.5.4 six==1.15.0 tensorboard==2.5.0 tensorboard-data-server==0.6.1 tensorboard-plugin-wit==1.8.0 tensorflow==2.4.0 tensorflow-estimator==2.4.0 termcolor==1.1.0 typing-extensions==3.7.4.3 urllib3==1.26.6 Werkzeug==2.0.1 wrapt==1.12.1 zipp==3.5.0 kspath

# Add the instruction to clone the repository, and then so we can access it
