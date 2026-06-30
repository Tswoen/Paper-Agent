# router 目录说明

本文档的目标不是简单列文件名，而是帮你建立一个稳定的阅读顺序：

1. `router` 这一层到底负责什么。
2. 每个文件在整个调用链里扮演什么角色。
3. 一次前端请求是怎么从 FastAPI 路由走到业务逻辑，再返回给前端的。
4. 哪些文件是“适配层”，哪些文件才是真正的“业务层”。

如果你之前觉得这部分“很乱”，核心原因通常不是代码量大，而是不同层的代码混在脑子里了。这个目录现在尽量只保留服务端相关代码与协议说明。

---

## 一、整体分层

这个目录目前可以理解成 5 层：

- 入口层：创建 FastAPI 应用并注册路由。
- 启动信息层：告诉前端当前运行能力，而不是做登录鉴权。
- 路由适配层：只负责把 HTTP 请求转成 Python 调用。
- 业务层：真正处理 settings、sessions、messages。
- 前端消费层：把后端事件整理成前端时间线。

---

## 二、主调用链

```text
main.py
  -> src/router/app.py:create_app()
       -> 注册 /webui/bootstrap
       -> 注册 /api/settings/*
       -> 注册 /api/sessions/*
       -> 注册 /api/sessions/{key}/messages

前端请求 /api/settings/*
  -> settings_router.py
  -> settings_api.py
  -> SettingsRepository

前端请求 /api/sessions/*
  -> sessions_router.py
  -> sessions_api.py
  -> SessionRepository

前端请求 /api/sessions/{key}/messages
  -> sessions_router.py
  -> sessions_api.py:submit_message
  -> SessionRepository
  -> 返回 events + thread

前端拿到 events
  -> stream_aggregator.py:ChatStreamAggregator
  -> 聚合成 UI 时间线 messages
```

---

## 三、为什么单机版去掉鉴权是合理的

当前项目的运行前提是：

- 用户把项目跑在自己的电脑上
- 不对外提供公共网页服务
- 没有多用户隔离需求

所以现在的设计改成了：

- 保留 `/webui/bootstrap`
- 去掉 token 签发
- 去掉所有 API 鉴权
- 直接允许本地前端访问 `/api/*`

这样更符合“单机版桌面工具”而不是“线上多用户服务”的定位。

---

## 四、逐文件说明

### 1. `app.py`

这是整个 `router` 目录最重要的文件，因为它是 FastAPI 应用的总装配点。

它主要做 4 件事：

1. 创建 `FastAPI` 对象。
2. 初始化仓库对象，比如 `SettingsRepository`、`SessionRepository`。
3. 注册设置相关路由和会话相关路由。
4. 统一处理异常转换。

关键函数：`create_app(...)`

### 2. `/webui/bootstrap` 逻辑

这部分逻辑现在已经并入 `app.py`，不再单独保留 `gateway.py`。

它存在的意义是：

- 给前端一个稳定的 bootstrap 入口
- 告诉前端当前运行能力
- 把启动相关配置集中在应用层

### 3. `settings_router.py`

这是“设置模块的 FastAPI 适配层”。

它只负责：

1. 解析 HTTP 参数。
2. 调业务函数。
3. 把 `SettingsError` 变成 JSONResponse。

### 4. `sessions_router.py`

这是“会话模块的 FastAPI 适配层”。

它负责这些接口：

- `GET /api/sessions`
- `POST /api/sessions`
- `GET /api/sessions/{session_key}/webui-thread`
- `DELETE /api/sessions/{session_key}`
- `POST /api/sessions/{session_key}/messages`

### 5. `settings_api.py`

这是设置业务核心。

它负责：

- settings 的读取
- settings 的更新
- provider/preset/defaults 的解析
- 返回前端完整快照

### 6. `sessions_api.py`

这是会话业务核心。

它负责：

- 创建会话
- 读取会话列表
- 读取线程
- 删除会话
- 持久化用户消息
- 处理消息提交流程并生成事件列表

### 7. `stream_aggregator.py`

这是给前端消费后端事件用的。

如果后端返回的是很多碎片事件，这个文件就负责把它们整理成前端真正想展示的消息时间线。

### 8. `protocol.py`

这个文件现在只保留前端时间线消息模型 `UIMessage`。

### 9. `__init__.py`

这是 `router` 包的统一导出入口。

### 10. `config_router.py`

这个文件目前本质上是 `__init__.py` 的重复导出版本，更像兼容层。

### 11. `front_to_back.md`

这是对前后端接口的补充说明文档，不是运行时代码。

### 12. `README.md`

也就是你现在正在看的这个文件。

---

## 五、补充说明

原来 `src/router/frontend_api.py` 里有一个“前端 API 客户端的 Python 版”。

但它并不参与服务端运行，只是测试里为了更方便模拟前端调用而存在。为了避免把测试工具和服务端代码混在一起，这个客户端已经迁移到了：

- `test/frontend_api_client.py`

所以现在 `src/router` 目录只保留服务端实现和协议说明，不再放测试辅助客户端。
