FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ADD ./ /workspace
WORKDIR /workspace
RUN apt update && apt install -y pip
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt