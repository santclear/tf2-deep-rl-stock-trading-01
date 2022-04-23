

### The rl_trader_tf2.py project is a copy, with some minor modifications made by me to work on TF 2.8 and increase training performance, presented in the course (Tensorflow 2.0: Um Guia Completo sobre o novo TensorFlow - Udemy) by Professor Jones Granatyr.

#### How to install the libraries in Anaconda

**First Method - Recommended if you are on Windows**

* conda create -n tf-gpu-cuda tensorflow-gpu cudatoolkit
* conda activate tf-gpu-cuda
* conda install -c conda-forge matplotlib
* conda install -c conda-forge pandas
* conda install -c conda-forge pandas-datareader
* conda install -c conda-forge tqdm
* conda install -c conda-forge spyder
* pip install yfinance

**Second Method - Recommended if you are on Linux (Note: I haven't tested it as my current OS is Windows, but I found it in tutorials on the web)** 

* conda create -n tf-gpu-cuda-nvidia
* conda activate tf-gpu-cuda-nvidia
* conda install -c nvidia cuda-toolkit
* conda install tensorflow-gpu -c conda-forge #https://conda-forge.org/blog/posts/2021-11-03-tensorflow-gpu/
* conda install -c conda-forge matplotlib
* conda install -c conda-forge pandas
* conda install -c conda-forge pandas-datareader
* conda install -c conda-forge tqdm
* conda install -c conda-forge spyder
* pip install yfinance

**Third Method - Recommended if you are on Linux (Note: I haven't tested it as my current OS is Windows, but I found it in tutorials on the web)**
* conda create -n tf-gpu-cuda-forge
* conda activate tf-gpu-cuda-forge
* conda install -c conda-forge cudatoolkit-dev
* conda install tensorflow-gpu -c conda-forge https://conda-forge.org/blog/posts/2021-11-03-tensorflow-gpu/
* conda install -c conda-forge matplotlib
* conda install -c conda-forge pandas
* conda install -c conda-forge pandas-datareader
* conda install -c conda-forge tqdm
* conda install -c conda-forge spyder
* pip install yfinance
