FROM  pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

RUN apt update
RUN apt-get install -y vim software-properties-common
RUN add-apt-repository -y ppa:jonathonf/ffmpeg-4
RUN apt-get update
RUN apt-get install -y libmediainfo-dev ffmpeg


WORKDIR /workspace
ADD . .
RUN pip install -r requirements.txt

RUN chmod -R a+w /workspace
RUN /bin/bash