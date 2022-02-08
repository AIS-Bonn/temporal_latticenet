FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,display,video

# Arguments
ARG user
ARG uid
ARG home
ARG workspace
ARG shell

# switch to non interactive https://github.com/phusion/baseimage-docker/issues/319
ENV DEBIAN_FRONTEND noninteractive

# copy the cleanup script so that its accesible in docker https://stackoverflow.com/a/53167819
COPY cleanup.sh ./
RUN chmod +x ./cleanup.sh

# Basic Utilities (dirmngr and gpg-agent is needed so that apt-get adv works)
RUN apt-get -y update && apt-get install -y --no-install-recommends sudo ssh tmux git meld cmake cmake-curses-gui htop vim iputils-ping apt-utils apt-transport-https software-properties-common wget build-essential gdb valgrind silversearcher-ag curl dirmngr gpg-agent checkinstall locate baobab ninja-build libncurses5-dev pkg-config && sh "./cleanup.sh"

RUN  echo "    IdentityFile ~/.ssh/id_rsa" >> /etc/ssh/ssh_config
RUN  echo "    IdentityFile ~/.ssh/id_ed25519" >> /etc/ssh/ssh_config

#-------------------------------------------------------------------------------

#python3
COPY cleanup.sh ./
RUN chmod +x ./cleanup.sh
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends python3-dev python3-pip python3-setuptools && sh "./cleanup.sh"

#switch to python3  https://linuxconfig.org/how-to-change-from-default-to-alternative-python-version-on-debian-linux
#this is needed so that catkin uses python3 and therefore pybind uses python3 which is the only one that can use spconv..
RUN sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2

#packages that were previously installed by ros but now we need the expliticly
RUN sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get -y install libopencv-dev libboost-all-dev libpcl-dev 


#GRAPHICS STUFF
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends libglfw3-dev && sh "./cleanup.sh"

RUN wget http://mirrors.edge.kernel.org/ubuntu/pool/universe/e/eigen3/libeigen3-dev_3.3.7-3_all.deb
RUN sudo dpkg -i libeigen3-dev_3.3.7-3_all.deb
RUN rm libeigen3-dev_3.3.7-3_all.deb

#pytorch
WORKDIR /
RUN git clone  https://github.com/pytorch/pytorch  /opt/pytorch
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends python3-pip python3-setuptools && sh "./cleanup.sh"
RUN pip3 install numpy pyyaml mkl mkl-include typing dataclasses
WORKDIR /opt/pytorch
RUN git checkout tags/v1.7.1
RUN git submodule sync
RUN git submodule update --init --recursive
ENV BUILD_TORCH ON 
ENV BUILD_CAFFE2_OPS OFF
ENV BUILD_CAFFE2_MOBILE OFF
ENV MAX_JOBS 4
RUN MAX_JOBS=8 USE_NINJA=1 BUILD_TEST=0 BUILD_JNI=0 TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0+PTX" python3 setup.py install 
WORKDIR /
# #need to make it own by myself and not by root because otherwise when we run python it will not access it
WORKDIR /
RUN git clone --recursive https://github.com/pytorch/vision /opt/vision
RUN pip3 install typing-extensions 
WORKDIR /opt/vision
RUN git checkout tags/v0.8.2
RUN git submodule sync
RUN git submodule update --init --recursive 
RUN MAX_JOBS=8 USE_NINJA=1 BUILD_TEST=0 BUILD_JNI=0 TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0+PTX" python3 setup.py install
WORKDIR /

#torch scatter in order to have scatter_max which is useful for a pointnet architecture. It needs the cuda_arch list in order to find nvidia drivers and build the gpu part of the torch-scatter https://github.com/rusty1s/pytorch_scatter/pull/79
#FORCE_CUDA may be needed for cases like this when the cuda runtime is not exposed during docker build but is actually needed https://github.com/facebookresearch/maskrcnn-benchmark/issues/167
ARG CUDA_HOME="/usr/local/cuda"
ARG TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.0+PTX"
ARG TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ARG CPATH="/usr/local/cuda/include" 
RUN git clone --recursive https://github.com/rusty1s/pytorch_scatter /opt/torch_scatter
WORKDIR /opt/torch_scatter
RUN git checkout tags/2.0.4
RUN git submodule sync
RUN git submodule update --init --recursive
RUN FORCE_CUDA=1 python3 setup.py install 
WORKDIR /

#python3 libs 
RUN sudo apt-get update && sudo apt-get -y install python3-pip python3-tk python3-pyqt5 python3-crypto python3-gnupg python3-dbg && sh "./cleanup.sh"
RUN sudo python3 -m pip install --upgrade pip
RUN sudo apt-get update && sudo python3 -m pip install ipdb numpy jupyter h5py scipy ipython pillow matplotlib opencv-python wheel trollius rospkg asyncio netifaces visdom torchnet sympy tqdm dlutils sklearn scikit-image twine natsort termcolor && sh "./cleanup.sh"
RUN sudo python3 -m pip install --upgrade setuptools setuptools-git  && sh "./cleanup.sh" 

RUN sudo pip install hjson torchviz plyfile

RUN apt-get -y update && apt-get install -y --no-install-recommends unzip && sh "./cleanup.sh"

WORKDIR /

#·-------------------------------------------------------------------------------

#Cleanup
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update
WORKDIR / 
RUN rm cleanup.sh

# Make SSH available
EXPOSE 22
EXPOSE 42421
# TensorBoard https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
EXPOSE 6006
# IPython https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
EXPOSE 8888

# Switch to the workspace
WORKDIR ${workspace}

RUN useradd -u ${uid} -ms /bin/bash ${user}

# Clone user into docker image and set up X11 sharing
RUN \
  echo "${user}:x:${uid}:${uid}:${user},,,:${home}:${shell}" >> /etc/passwd && \
  echo "${user}:x:${uid}:" >> /etc/group && \
  echo "${user} ALL=(ALL) NOPASSWD: ALL" > "/etc/sudoers.d/${user}" && \
  chmod 0440 "/etc/sudoers.d/${user}"


#Set the user in the approapriate groups
RUN usermod -a -G dialout ${user}
RUN usermod -a -G video ${user}
RUN usermod -a -G audio ${user}
RUN usermod -a -G plugdev ${user}

#own some stuff because they were created by the root user in the docker container
RUN chown -R ${user} /opt
USER ${user}
USER root

# Switch to user
RUN chown -R ${user} ${workspace}
# RUN echo ${workspace}
USER "${user}"

# switch to interactive
ENV DEBIAN_FRONTEND teletype