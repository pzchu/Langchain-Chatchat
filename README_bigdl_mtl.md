# Run on MTL iGPU with BigDL-LLM optimizations for native Windows

## Prerequisites
To benefit from BigDL-LLM GPU acceleration, there’re several prerequisite steps for tools installation and environment preparation.

Please refer to our [QuickStart guide](https://bigdl.readthedocs.io/en/latest/doc/LLM/Quickstart/install_windows_gpu.html) and follow step [Install Visual Studio 2022](https://bigdl.readthedocs.io/en/latest/doc/LLM/Quickstart/install_windows_gpu.html#install-visual-studio-2022), [Install GPU Driver](https://bigdl.readthedocs.io/en/latest/doc/LLM/Quickstart/install_windows_gpu.html#install-gpu-driver), and [Install oneAPI](https://bigdl.readthedocs.io/en/latest/doc/LLM/Quickstart/install_windows_gpu.html#install-oneapi).

We also recommend using Miniconda for environment management. Refer to [Miniconda Installation page](https://docs.anaconda.com/free/miniconda/) to install for Windows.

## Create Environment
Use the following command in **Anaconda Prompt (miniconda3)** to create the environment:

```cmd
git clone https://github.com/intel-analytics/Langchain-Chatchat.git
cd Langchain-Chatchat

conda create -n bigdl-langchain-chatchat python=3.11 libuv 
conda activate bigdl-langchain-chatchat

pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install --pre --upgrade torchaudio==2.1.0a0  -f https://developer.intel.com/ipex-whl-stable-xpu

pip install -r requirements_bigdl.txt 
pip install -r requirements_api_bigdl.txt
pip install -r requirements_webui.txt
```

## Start the Service
### Prepare the configuration file
```bash
python copy_config_example.py
```
And then change `MODEL_ROOT_PATH` in `configs\model_config.py` to the folder path where you place the models (LLMs, embedding models, etc.)

> Note that currently only the default [chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b) and [bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) has been verified. For services with default model settings, please make sure you have these two models in your `MODEL_ROOT_PATH` folder.

### Start the service
In the Anaconda Prompt windows:
```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
set SYCL_CACHE_PERSISTENT=1
set BIGDL_LLM_XMX_DISABLED=1
set no_proxy=localhost,127.0.0.1
python startup.py -a
```
And the service will start at http://localhost:8501/