# Run on MTL iGPU with BigDL-LLM optimizations for native Windows

## Prerequisites
To benefit from BigDL-LLM GPU acceleration, there’re several prerequisite steps for tools installation and environment preparation.

Please refer to our [QuickStart guide](https://bigdl.readthedocs.io/en/latest/doc/LLM/Quickstart/install_windows_gpu.html) and follow step [Install Visual Studio 2022](https://bigdl.readthedocs.io/en/latest/doc/LLM/Quickstart/install_windows_gpu.html#install-visual-studio-2022), [Install GPU Driver](https://bigdl.readthedocs.io/en/latest/doc/LLM/Quickstart/install_windows_gpu.html#install-gpu-driver), and [Install oneAPI](https://bigdl.readthedocs.io/en/latest/doc/LLM/Quickstart/install_windows_gpu.html#install-oneapi).

We also recommend using Miniconda for environment management. Refer to [Miniconda Installation page](https://docs.anaconda.com/free/miniconda/) to install Miniconda for Windows.

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

## Prepare Configuration Files
```bash
python copy_config_example.py
```
And then change `MODEL_ROOT_PATH` in `configs\model_config.py` to the folder path where you will place the models (LLMs, embedding models, etc.)

### Download models
Currently, LLMs `THUDM/chatglm3-6b` and `meta-llama/Llama-2-7b-chat-hf`, as well as embedding model `BAAI/bge-large-zh-v1.5` have been supported. Please download these 3 models to your `MODEL_ROOT_PATH` folder and **change the folder name of your downloaded model as required**:

| Model | link to download | Downloaded model folder name |
|:--|:--|:--|
|`THUDM/chatglm3-6b`| [HF](https://huggingface.co/THUDM/chatglm3-6b) or [ModelScope](https://www.modelscope.cn/models/ZhipuAI/chatglm3-6b/summary) | chatglm3-6b |
|`meta-llama/Llama-2-7b-chat-hf`| [HF](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | bigdl-7b-chat-hf |
|`BAAI/bge-large-zh-v1.5`| [HF](https://huggingface.co/BAAI/bge-large-zh-v1.5) | bge-large-zh-v1.5 |

## One-time Warmup
It is required to conduct a one-time warmup for GPU kernels compilation.

In the Anaconda Prompt windows:
```cmd
python warmup.py
```

> The warmup may take several minutes.

## Start the Service
In the Anaconda Prompt windows:
```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
set SYCL_CACHE_PERSISTENT=1
set BIGDL_LLM_XMX_DISABLED=1
set no_proxy=localhost,127.0.0.1
python startup.py -a
```
And the service will start at http://localhost:8501/