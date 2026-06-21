unittest 是 Python 标准库自带，无需额外 pip 安装


# 命令完整拆解：`uv run python -m unittest discover -s test -v`
## 一、分段逐个解释
### 1. `uv run`
`uv` 是 Python 新一代包管理/虚拟环境工具（替代 pip、venv、poetry）
- `uv run`：**在当前项目隔离虚拟环境中执行后面的命令**
  自动读取项目 `pyproject.toml` 依赖，保证使用本项目安装的包，不会污染全局 Python；
  等价于手动激活 venv 后再运行 python，但更简洁。

### 2. `python`
使用 uv 环境内绑定的 Python 解释器运行程序。

### 3. `-m unittest`
以模块方式运行 Python 内置标准单元测试库 `unittest`，不用手动写测试入口脚本。

### 4. `discover`
unittest 的自动发现子命令：自动扫描并收集所有测试用例，规则：
1. 默认递归查找目录下命名匹配 `test*.py` 的文件；
2. 文件内以 `Test` 开头的类、以 `test_` 开头的方法都会被当作测试用例执行。

### 5. `-s test`
`-s` = `--start-directory`，指定**测试扫描根目录**为项目下的 `test/` 文件夹；
不写 `-s` 默认从当前目录 `.` 开始找，这里限定只去 `test/` 目录搜测试文件。

### 6. `-v`
`-v` = verbose，详细输出模式：
- 不加 `-v`：只显示成功/失败总数；
- 加 `-v`：逐条打印每个执行的测试方法名、成功/失败标记，方便定位出错用例。

## 二、整条命令功能
使用 uv 管理的项目虚拟环境，执行 Python 内置 unittest 测试框架，**自动扫描 `test/` 目录下所有测试文件并完整打印每条测试的执行详情**。

## 三、补充常用配套参数
1. 只跑单个测试文件
```bash
uv run python -m unittest test/test_llm.py -v
```
2. 限制只跑某一个测试类/方法
```bash
uv run python -m unittest test.test_llm.TestLLMClient.test_chat_stream -v
```
3. 失败时立刻停止不再执行后续用例
```bash
uv run python -m unittest discover -s test -v -f
```

4. 运行真实模型联调测试（会读取 `config/model.json` 并实际请求模型）
```bash
$env:RUN_LIVE_MODEL_TEST="1"
uv run python -m unittest test.test_live_model_config -v
```
这个测试默认跳过，避免日常跑单测时误调用外部模型并消耗额度。

## 四、对比说明
如果你不用 uv，原生等价命令：
```bash
python -m unittest discover -s test -v
```
区别仅在于 `uv run` 会锁定项目依赖环境，避免全局包版本不一致导致测试报错。
