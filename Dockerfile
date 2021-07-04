FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update

# for opencv-python
RUN apt-get install -y libgl1-mesa-glx

RUN apt-get install -y vim

RUN apt-get install -y libgtk2.0-dev

COPY ./requirements.txt ./requirements.txt 

RUN pip install -r requirements.txt