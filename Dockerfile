# Use the official Miniconda base image
FROM continuumio/miniconda3:24.3.0-0

# Set the working directory
WORKDIR /workspace

# Copy your project files
COPY . /workspace

# Create the environment
# RUN conda env create -f minimal.yml
RUN conda env create -f environment.yml && \
    conda clean --all -y

#install the library
RUN conda run -n env_fastai pip install seqdata/. rarfile easyDataverse

#prepare datasets in container in /local_data for easy access
RUN conda run -n env_fastai python 0_prepare_datasets.py

# Activate the environment by default
RUN echo "source activate env_fastai" >> ~/.bashrc


# Start jupyter lab with the correct conda environment by default
EXPOSE 8888
ENTRYPOINT ["bash", "-c", "source activate env_fastai && exec \"$@\"", "--"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
