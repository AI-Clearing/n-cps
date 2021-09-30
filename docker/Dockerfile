FROM nvidia/cuda:10.2-devel-ubuntu18.04
# FROM charlescxk/ssc:2.0
# Inspired by https://towardsdatascience.com/conda-pip-and-docker-ftw-d64fe638dc45

# Change shell to bash (sh is not supported by conda)
SHELL [ "/bin/bash", "--login", "-c" ]

# Create a non-root user
ARG username=ubuntu
ARG uid=1234
ARG gid=100
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER
RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER


RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get install -y wget libglib2.0-0 libsm6 libxrender1 libxext6
RUN apt-get install -y 


COPY semiseg.yaml $HOME/semiseg.yaml
RUN chown $UID:$GID $HOME/semiseg.yaml

USER $USER
# install miniconda
ENV MINICONDA_VERSION py39_4.9.2
ENV CONDA_DIR $HOME/anaconda3

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH
# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

# create a project directory inside user home
ENV PROJECT_DIR $HOME/ncps
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

# build the conda environment
ENV CONDA_ENV semiseg
RUN conda update --name base --channel defaults conda --yes && \
    conda env create --file $HOME/semiseg.yaml --force && \
    conda clean --all --yes
# run the postBuild script to install any JupyterLab extensions
# RUN conda activate semiseg

USER root

# Copy and build Apex (will need GPU for that!)
RUN mkdir apex
COPY ./furnace/apex/ $PROJECT_DIR/apex/
RUN conda activate $CONDA_ENV && cd ${PROJECT_DIR}/apex/ && python setup.py install --cpp_ext --cuda_ext && cd ${PROJECT_DIR}

# Copy the project
COPY ./ $PROJECT_DIR/
RUN chown -R $UID:$GID $PROJECT_DIR

COPY docker/entrypoint.sh /usr/local/bin/
RUN chown $UID:$GID /usr/local/bin/entrypoint.sh && \
    chmod u+x /usr/local/bin/entrypoint.sh
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]

# USER $USER
