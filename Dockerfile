FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime
MAINTAINER Taekmin Kim <tantara.tm@gmail.com>

RUN pip install cython cffi opencv-python scipy easydict matplotlib pyyaml tqdm
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
RUN apt-get update --fix-missing
RUN apt-get install -y screen htop git vim

ENV APP_PATH /base
RUN mkdir -p $APP_PATH
WORKDIR $APP_PATH

RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:pytorchkr' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
