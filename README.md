# Qwen2.5-Omni-7B
Run Qwen2.5-Omni-7B with ipex-llm on Intel platform

# 模型下载
推荐使用魔搭下载到本地Qwen2.5-Omni-7B 目录

# 环境搭建
推荐使用minconda或者anaconda创建虚拟环境，请选择合适的ipex-llm安装指令
请参考 https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/install_pytorch26_gpu.md

该指南给出的是运行在ARL-H平台的参考运行流程：

搭建ipex-llm环境
  <pre> 
  conda create -n llm-pt26 python=3.11
  conda activate llm-pt26
  pip install --pre --upgrade ipex-llm[xpu_2.6_arl] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/arl/cn/ </pre>

跟新模型相关需求
  <pre> 
  pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
  pip install accelerate==1.5.2
  pip install qwen-omni-utils 
  pip install pandas </pre>

设置环境变量：
  <pre> 
  set SYCL_CACHE_PERSISTENT=1
  set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 </pre>
  


