## 一、mindrlhf运行Qwen2.5步骤

### 1. 模型以及数据集获取与预处理 以及版本对齐

####  1.1 版本对齐

**vllm**

```text
https://repo.mindspore.cn/mirrors/vllm/version/202505/20250514/v0.8.4.dev0_newest/any/vllm-0.8.4.dev0%2Bg296c657.d20250514.empty-py3-none-any.whl
```

**vllm-mindspore**

```text
https://repo.mindspore.cn/mindspore/vllm-mindspore/version/202508/20250807/r0.3.0_20250807104902_4739c2e599777d7790b444ec1fb27573fb941002_newest/ascend/aarch64/

https://repo.mindspore.cn/mindspore/vllm-mindspore/version/202508/20250807/r0.3.0_20250807104902_4739c2e599777d7790b444ec1fb27573fb941002_newest/ascend/aarch64/vllm_mindspore-0.3.0-cp310-cp310-linux_aarch64.whl
```

**msadapter**

```text
https://repo.mindspore.cn/mindspore/msadapter/version/202508/20250807/r0.2.0_20250807013007_e7636d61563c4beafac4b877891172464fdcf321_newest/any/

https://repo.mindspore.cn/mindspore/msadapter/version/202508/20250807/r0.2.0_20250807013007_e7636d61563c4beafac4b877891172464fdcf321_newest/any/msadapter-0.0.1-py3-none-any.whl
```

**mindspore_gs**

```text
https://repo.mindspore.cn/mindspore/golden-stick/version/202506/20250604/master_20250604160014_35fcbec4406d3b18faf02ef99fcbe2741e80348e_newest/any/

https://repo.mindspore.cn/mindspore/golden-stick/version/202506/20250604/master_20250604160014_35fcbec4406d3b18faf02ef99fcbe2741e80348e_newest/any/mindspore_gs-1.2.0.dev20250604-py3-none-any.whl
```

**mindspore**

```text
https://repo.mindspore.cn/mindspore/mindspore/version/202508/20250807/r2.7_20250807154652_7edec76ede691ac90be9590b0ebb2a65923b55fe_newest/unified/aarch64/

https://repo.mindspore.cn/mindspore/mindspore/version/202508/20250807/r2.7_20250807154652_7edec76ede691ac90be9590b0ebb2a65923b55fe_newest/unified/aarch64/mindspore-2.7.0-cp310-cp310-linux_aarch64.whl
```

**mindformers**

```text
git checkout 051d65f
```



#### 1.2 获取源代码

```shell
git clone https://gitee.com/mindspore/mindrlhf.git
bash build.sh


git clone https://gitee.com/mindspore/mindformers.git
cd mindformers
git checkout 051d65f
git rev-parse HEAD  # 用于检查版本  git status
bash build.sh
```



#### 1.3 设置 msprobe 

不然后面会报错：识别不到'save'

```shell
git clone https://gitcode.com/Ascend/mstt.git
cd mstt/debug/accuracy_tools

pip install setuptools wheel

python setup.py bdist_wheel --include-mod=[adump]
cd ./dist
pip install ./mindstudio_probe*.whl
```





### 2. GRPO

#### 2.1 数据集获取和预处理

获取数据集并将其转换为`.mindrecord`格式，这里用的是mindrlhf里面的`rlhf_data.py`程序

```shell
# 建议新开个terminal

# 获取模型
mkdir models
cd models
mkdir Qwen2.5

pip install modelscope

modelscope download --model Qwen/Qwen2.5-7B-Instruct  --local_dir   /home/ma-user/work/models/Qwen2.5-Instruct

mkdir data
cd data
# 下载gsm8k数据集
git clone https://github.com/openai/grade-school-math.git

# 下载完成后，需要转为MindSpore使用的.mindrecord文件   首先进入MindRLHF路径 并执行以下脚本：
cd mindrlhf

pip install jsonlines

export PYTHONPATH=/home/ma-user/work/mindformers:$PYTHONPATH

python examples/grpo/qwen_grpo_tutorial/rlhf_data.py \
--vocab_path /home/ma-user/work/models/Qwen2.5-Instruct/vocab.json \
--merges_file_path /home/ma-user/work/models/Qwen2.5-Instruct/merges.txt \
--file_path /home/ma-user/work/data/grade-school-math/grade_school_math/data/train.jsonl \
--output_path /home/ma-user/work/data/gsm8k_train.mindrecord \
--dataset_type gsm8k
# 此脚本会将train.jsonl转换成mindrecord的形式保存在/{path}/gsm8k_train.mindrecord。此数据路径将在训练拉起时作为mind_dataset_dir的值被传入。
```



#### 2.2 bash run_grpo_vllm.sh来启动vllm加速

修改`examples/grpo/qwen_grpo_tutorial/run_grpo_vllm.sh`中的配置路径，bash运行即可

```shell
#!/bin/bash

export MS_ALLOC_CONF="memory_tracker:False,enable_vmm:True"
export GLOG_v=2

export vLLM_MODEL_BACKEND=MindFormers
export HCCL_EXEC_TIMEOUT=7200
export MS_JIT_MODULES=vllm_mindspore,research
export MS_ENABLE_LCCL=off

root_path="$(realpath "$(dirname "$0")")"
root_path=$root_path/../../../
cd $root_path
export PYTHONPATH=$root_path:$PYTHONPATH  # define mindrlhf path

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest # need modify
export MINDFORMERS_PATH=/home/ma-user/work/mindformers # need modify
export MSADAPTER_PATH=/path/to/msadapter # need modify (msadapter lib path)
export QWEN_MODEL_PATH=/home/ma-user/work/models/Qwen2.5-Instruct
export DATASET_FILE=/home/ma-user/work/data/gsm8k_train.mindrecord
export SAVE_CHECKPOINT_DIR=/home/ma-user/work/mindrlhf/output/grpo_checkpoints

export PYTHONPATH=$MSADAPTER_PATH:$MINDFORMERS_PATH:$PYTHONPATH


msrun --worker_num=8 --local_worker_num=8 \
--master_addr=127.0.0.1 --master_port=9887 \
--join=True --log_dir=./prof_vllm_log \
examples/grpo/qwen_grpo_tutorial/main.py \
--config examples/grpo/qwen_grpo_tutorial/grpo_config.yaml \
--tokenizer_dir $QWEN_MODEL_PATH \
--dataset_file $DATASET_FILE \
--save_checkpoint_dir $SAVE_CHECKPOINT_DIR \
--actor_checkpoint_path $QWEN_MODEL_PATH \
--ref_checkpoint_path $QWEN_MODEL_PATH \
--generate_checkpoint_path $QWEN_MODEL_PATH \
--verifier_function "qwen_accuracy_reward,format_reward" \
--verifier_weight "1.0,1.0" > vllm.log 2>&1 &
```



## 3. 一键配置环境脚本

在`work`目录下 下载好包的.whl文件后，bash即可

```shell
pip install vllm-0.8.4.dev0+g296c657.d20250514.empty-py3-none-any.whl

pip3 uninstall torch torch-npu torchvision

pip install vllm_mindspore-0.3.0-cp310-cp310-linux_aarch64.whl

pip install msadapter-0.0.1-py3-none-any.whl

pip install mindspore_gs-1.2.0.dev20250604-py3-none-any.whl

pip install mindspore-2.7.0-cp310-cp310-linux_aarch64.whl

pip install ray


# 解决一直遇到的bug
pip install --upgrade 'urllib3==1.26.7' 

# 安装mindformers
cd mindformers
git checkout 051d65f
bash build.sh

# 安装mindrlhf
cd ../mindrlhf
bash build.sh

# 安装mindstudio_probe 避免'save'报错

cd ../ #回到work目录下
git clone https://gitcode.com/Ascend/mstt.git

cd mstt/debug/accuracy_tools

pip install setuptools wheel

python setup.py bdist_wheel --include-mod=[adump]
cd ./dist
pip install ./mindstudio_probe*.whl
```

























































