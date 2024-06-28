FROM python:3.10

# Set the working directory
WORKDIR /workspace

# Copy your project files
COPY . /workspace

# install latex for matplotlib
RUN apt-get update && apt-get install -y \
    texlive-latex-extra \
    texlive-fonts-recommended \
    dvipng \
    cm-super \
    && rm -rf /var/lib/apt/lists/*

#install pytorch with cuda 11.5
RUN pip install torch==1.11 --index-url https://download.pytorch.org/whl/cu115  --no-cache-dir

#install the library
RUN pip install seqdata/. jupyterlab h5py ipympl seaborn rarfile easyDataverse --no-cache-dir

#prepare datasets in container in /local_data for easy access
RUN python 0_prepare_datasets.py

# Start jupyter lab with the correct conda environment by default
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
#jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser