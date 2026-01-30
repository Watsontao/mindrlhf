代码学习笔记

```python
字典 dict	    {'a': 1, 'b': 2} 或 dict(a=1)     一对一 键唯一
元组 tuple	(1, 2, 3) 或 1,（单元素需逗号）      一对一 不可修改 可重复
列表 list	    [1, 2, 3] 或 list()               有序 可重复
集合 set	    {1, 2, 3} 或 set()                无序 自动去重

runner = task_runner_class.remote()
ray.get(runner.run.remote(config))  # runner.run.remote(config)   这是一个异步调用，主进程发出指令后，立刻返回一个“回执单”（在 Ray 里叫 ObjectRef 或 Future），不会等 run 函数跑完。在 Ray 中，调用 Actor 的方法必须加 .remote(...)。  
# ray.get() 是一个阻塞调用
ray.remote(...) #把一个普通的 Python 类转换成了 Ray Actor 类。这意味着当你后面调用 Role.Actor.remote() 时，Ray 会在集群的某个远程节点上启动一个全新的进程来运行这个类


pprint(OmegaConf.to_container(config, resolve=True)) # resolve=True 表示解析配置中的插值变量（例如 ${path.home} 会变成具体的路径 /home/user

resolve(config): 直接在原对象上进行解析，确保后续访问 config 时所有的变量引用都已经有了确定的值。
copy_to_local：如果模型路径是在 HDFS 或 S3 上，先把它下载到本地磁盘。如果本来就是本地路径，就直接返回路径

config 整体不是普通 dict，而是 OmegaConf 对象（通常是 DictConfig 实例）。
但因为它“长得像 dict”，所以：
支持 config.data 这种属性式访问；
也支持 config["data"] 这种字典式访问；
同时也实现了 .get(key, default) 方法，行为与 dict.get 一致。

trust_remote_code： 是否允许加载 Hugging Face 仓库里自带的自定义代码（tokenizer.py），然后真正实例化分词器
“tokenizer 只管文字；processor 是多模态‘全家桶’，文字+图片+音频一起搞定。”

use_fast=True
强制使用 Rust 版快速分词器（tokenizers 库）作为底层 tokenizer；
如果该模型没有 fast 版本，HF 会自动退回到 Python 慢版本并给出提示。

self._create_rollouter(config) # 前置单下划线 _ 是 Python 社区约定：“这是内部实现，外部最好别直接调”——语法上完全公开，只是提醒。

FullyAsyncRollouter.remote(...)：#这是 Ray 的标准语法。它告诉 Ray 集群：“请找一台（或一组）有资源的机器，启动一个独立的进程来运行 FullyAsyncRollouter 类”。这个进程独立于当前的 Runner 进程，拥有自己的内存空间和事件循环。

在 Ray 中，所有的 .remote() 调用都是异步的。 因此要想等待 需要加ray.get()
.remote() 立即返回 ObjectRef——只是“任务已提交”的凭证，不保证远端方法已跑完。
套上 ray.get 后，主进程会阻塞直到远端任务执行完毕并返回，


@ray.remote(num_cpus=1) # 告诉 Ray：这个类不是普通类，而是远程 Actor；每创建一个 FullyAsyncTaskRunner Actor，必须给它留 1 颗 CPU 核心才启动；如果集群当前空闲 CPU 不足，就排队等。
class FullyAsyncTaskRunner:
    
    
    
runner = task_runner_class.remote() #.remote() 返回一个“远程句柄” ， 
# 如果Actor 类 .remote(...) → 返回 ActorHandle（代理对象）；
# 如果Actor 方法 .remote(...) → 返回 ObjectRef（Future）

ray.wait：这是 Ray 的核心调度原语。它能够高效地监控一组异步任务的状态。
asyncio.create_task(...)  # 把 协程对象（async def 函数调用后返回的 coroutine）包装成 Task，马上调度进事件循环，不阻塞当前线程。
asyncio.gather(...)
# 把多个协程/Task 打包成一个大任务，并发执行，最先完成的先回，最终返回一个列表，顺序与传入顺序一致。
# 默认只要其中任何一个抛异常，就会立即把异常抛给调用者，取消剩余任务。
return_exceptions=True # 改成“异常也当成普通结果”放进列表，不中断其他任务。这样你可以自己检查哪个任务失败了，而不会导致整个程序崩溃。

==========================
task.cancel() 的本质就是：在任务下一次“醒来”（遇到 await）的时候，往它身体里注入一个 asyncio.CancelledError 异常。

它不是操作系统的 kill -9（强制拔电源），而是一种**“礼貌的终止请求”**。

为了让你彻底理解，我们可以把这个过程拆解为三步：

1. 埋雷（Call cancel）
当你调用 generation_task.cancel() 时：

Python 并没有立马杀掉这个任务。

Python 只是在任务的日程表上标记了一下：“下次轮到这小子运行的时候，别让他继续跑了，直接给他抛个 CancelledError。”

2. 引爆（Next await）
当那个任务运行到下一行 await 代码（比如 await run_model()）时：

事件循环（Event Loop）准备恢复它。

突然发现它被标记了“取消”。

于是，await 语句不再返回正常结果，而是直接抛出 CancelledError 异常。

3. 善后（Cleanup）
这就解释了为什么需要最后的 await：

因为异常抛出来了，任务内部的 try...finally 块开始执行（比如关闭文件、释放显存）。

这个善后过程是需要花时间的。

如果你不 await 它，主程序直接退出了，可能导致善后工作做到一半就被掐断了（比如显存没释放干净）。
==========================
```



