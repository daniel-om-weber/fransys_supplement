# Use the official Miniconda base image
FROM python:3.10-alpine

# Set the working directory
WORKDIR /workspace

# Copy your project files
COPY . /workspace

#install the library
RUN pip install seqdata/. jupyterlab h5py ipympl seaborn rarfile easyDataverse

#prepare datasets in container in /local_data for easy access
RUN python 0_prepare_datasets.py

# Start jupyter lab with the correct conda environment by default
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
