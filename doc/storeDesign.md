# 会话存储系统设计文档

## 1. 背景

当前项目已经具备基础的会话能力：

- 前端可以创建会话、拉取会话列表、获取线程、向某个会话提交消息。
- 后端通过 `src/router/sessions_api.py` 中的 `SessionRepository` 保存会话状态。
- 后端通过 `src/router/realtime.py` 中的 `HttpMessageGateway` 生成一轮执行的事件列表。
- 前端通过 `src/router/stream_aggregator.py` 把事件列表聚合成可展示的时间线。

但是现在的 `SessionRepository` 仍然是“纯内存仓库”，意味着：

- 服务重启后历史会话会丢失。
- 执行过程只有最终线程，没有完整、可追溯的执行日志。
- 用户无法稳定地加载历史执行细节。
- 如果未来引入更复杂的 Agent、工具调用、文件编辑、附件或长时间任务，现有结构很快会失去扩展性。

因此，这个系统需要一个真正的“会话持久化存储系统”，既能保存最终对话内容，也能保存执行过程本身。

---

## 2. 设计目标

### 2.1 核心目标

本次存储系统设计要满足以下目标：

1. 支持会话持久化，服务重启后历史会话仍可加载。
2. 支持完整记录一次任务执行过程，而不只是最终 `messages`。
3. 支持用户快速查看会话列表与完整线程内容。
4. 支持恢复中断任务的执行状态，至少能够识别“执行中断”并安全回放。
5. 支持后续扩展到工具调用、文件编辑记录、附件、多轮任务和更复杂的 Agent 图执行。
6. 尽量不破坏现有前后端接口形状，使改造可以分阶段完成。

### 2.2 非目标

以下内容不作为第一版存储系统的强制目标：

- 多用户权限隔离
- 云端共享会话
- 分布式数据库部署
- 超大规模日志分析平台
- 对所有中间 token 级别增量做永久精确保存

也就是说，这一版优先服务“单机、本地、单用户、可恢复、可审计、可扩展”的工程目标。

---

## 3. 为什么不能只存 `messages`

如果只存现在的 `record.messages`，会有几个明显问题：

1. 无法区分“用户消息”“助手最终输出”和“执行过程中的中间事件”。
2. 无法回放 reasoning、tool、file_edit、goal_status 等事件。
3. 无法知道某轮任务什么时候开始、什么时候结束、是否异常中断。
4. 无法准确支持未来的“恢复执行”“任务审计”“导出完整轨迹”。
5. 当前前端展示的 `messages` 其实是事件聚合后的结果，不应该直接等同于底层事实数据。

所以底层存储不能只保存“展示结果”，还必须保存“过程事实”。

这意味着底层设计应采用：

- 事件日志作为事实源
- 消息/线程作为投影视图
- 快照作为快速恢复手段

---

## 4. 总体设计思路

### 4.1 总体原则

会话存储系统采用“三层数据结构”：

1. 事实层：追加写入的事件日志
2. 视图层：面向前端读取的会话摘要、消息线程、任务状态投影
3. 快照层：用于快速恢复和重建的聚合快照

### 4.2 总体架构

```text
前端
  -> FastAPI Router
  -> SessionApplicationService
  -> SessionStore(抽象接口)
  -> SQLite 会话数据库
  -> 本地文件资产目录

写路径:
提交消息
  -> 创建 turn
  -> 记录事件日志
  -> 增量更新线程投影
  -> 更新会话摘要
  -> 必要时写快照

读路径:
会话列表
  -> 读取会话摘要投影

线程详情
  -> 读取消息投影 + 最新运行状态

审计/调试/恢复
  -> 读取 turn + event 日志 + snapshot
```

### 4.3 为什么选择 SQLite 作为第一落地方案

当前项目明确偏向本地单机运行，因此第一版推荐使用 SQLite，而不是直接上 PostgreSQL。原因如下：

- 单机模式下部署成本最低。
- 持久化能力足够，支持事务、一致性、索引、JSON 文本字段。
- Python 集成简单，后续可替换为其他存储实现。
- 更符合“先让功能稳定可用，再扩展到更复杂部署”的路线。

但在架构设计上，不能把业务逻辑写死在 SQLite 上，而应该抽象为 `SessionStore` 接口。

---

## 5. 存储目录布局

建议默认使用独立的数据目录，而不是混在 `logs/` 中。

推荐目录结构如下：

```text
data/
  session_store/
    sessions.db
    artifacts/
      session_{session_id}/
        media/
        diffs/
        exports/
    backups/
```

推荐默认配置：

- 根目录：`./data/session_store`
- 数据库文件：`./data/session_store/sessions.db`
- 附件目录：`./data/session_store/artifacts`
- 通过配置项或环境变量覆盖，例如 `PAPERS_AGENT_STORE_DIR`

这样做的好处是：

- 会话数据与日志、配置、源码分离
- 容易做备份与清理
- 后续导出/导入时边界清晰

---

## 6. 核心存储模型

第一版建议至少包含 6 类核心对象：

1. `session`：会话主记录
2. `session_turn`：每一轮任务执行记录
3. `session_event`：事实事件日志
4. `session_message`：给前端直接读取的线程投影
5. `session_snapshot`：恢复快照
6. `session_asset`：附件、文件编辑产物、导出文件等引用

下面逐一展开。

### 6.1 `session` 会话主表

用途：

- 保存会话级元数据
- 支撑会话列表页
- 保存当前会话状态摘要

建议字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | TEXT | 会话主键，沿用现有 `session_key` |
| `title` | TEXT | 会话标题 |
| `created_at` | TEXT | 创建时间，UTC ISO8601 |
| `updated_at` | TEXT | 最近更新时间 |
| `status` | TEXT | `idle/running/failed/interrupted/deleted` |
| `last_turn_id` | TEXT | 最近一轮任务 ID |
| `last_message_preview` | TEXT | 会话列表预览文本 |  不理解
| `workspace_scope_json` | TEXT | 当前工作区范围 JSON |
| `latest_snapshot_id` | TEXT | 最新快照 ID |  不理解
| `archived_at` | TEXT NULL | 归档时间 |
| `deleted_at` | TEXT NULL | 软删除时间 |
| `version` | INTEGER | 乐观并发版本号 |  # 不需要

说明：

- `status` 不表示整个系统永久状态，而是“当前对外展示的会话状态摘要”。
- `last_message_preview` 用于会话列表，不需要每次重建整条线程。
- `version` 用于避免并发更新时覆盖旧状态。

### 6.2 `session_turn` 回合表

用途：

- 表示一次“用户发起任务 -> Agent 执行 -> 输出结果”的完整生命周期
- 支撑恢复、审计和运行态判断

建议字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | TEXT | 回合主键，对应 `turn_id` |
| `session_id` | TEXT | 所属会话 |
| `turn_seq` | INTEGER | 会话内递增序号 |
| `trigger_role` | TEXT | 发起者，通常为 `user` |
| `input_message_id` | TEXT | 触发该轮的用户消息 ID |
| `status` | TEXT | `queued/running/completed/failed/interrupted/cancelled` |
| `started_at` | TEXT | 开始时间 |
| `heartbeat_at` | TEXT NULL | 心跳时间 |
| `finished_at` | TEXT NULL | 结束时间 |
| `error_code` | TEXT NULL | 错误码 |
| `error_message` | TEXT NULL | 错误说明 |
| `executor_type` | TEXT | 执行器类型，如 `http_gateway` |
| `runtime_meta_json` | TEXT | 运行时元数据，如模型、节点、路由信息 |

说明：

- `heartbeat_at` 用于检测“服务异常退出导致未完成 turn 悬空”的情况。
- `runtime_meta_json` 用于保存当时所用模型、图执行器、路由策略、请求上下文等。

### 6.3 `session_event` 事件表   #不理解

用途：

- 作为事实源，记录执行过程中的所有重要事件
- 支撑回放、调试、审计、重建投影

建议字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | TEXT | 事件主键 |
| `session_id` | TEXT | 所属会话 |
| `turn_id` | TEXT | 所属回合 |
| `event_seq` | INTEGER | 回合内严格递增序号 |
| `event_type` | TEXT | 事件类型 |
| `role` | TEXT NULL | 角色，如 `user/assistant/system/tool` |
| `message_id` | TEXT NULL | 如果事件归属于某条消息，则写入 |
| `payload_json` | TEXT | 事件载荷 |
| `created_at` | TEXT | 创建时间 |
| `chunk_index` | INTEGER NULL | 用于流式分块 |
| `is_compacted` | INTEGER | 是否已被压缩归并 |

设计重点：

- `event_seq` 不能依赖时间排序，必须使用严格递增序号。
- `payload_json` 存结构化数据，避免未来事件类型扩展时频繁改表。
- 事件表是“最重要的事实源”，任何消息线程都应该可以由事件日志重建。

### 6.4 `session_message` 消息投影表

用途：

- 面向前端快速读取 `webui-thread`
- 保存聚合后的消息时间线，而不是所有原始碎片

建议字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | TEXT | 消息主键 |
| `session_id` | TEXT | 所属会话 |
| `turn_id` | TEXT | 所属回合 |
| `message_seq` | INTEGER | 会话内消息顺序 |
| `role` | TEXT | `user/assistant/system` |
| `kind` | TEXT | `message/progress/tool/...` |
| `content` | TEXT | 聚合后的正文 |
| `reasoning` | TEXT | 聚合后的 reasoning 文本 |
| `is_streaming` | INTEGER | 是否仍处于流式中 |
| `reasoning_streaming` | INTEGER | reasoning 是否仍流式中 |
| `tool_events_json` | TEXT | 与消息关联的工具事件数组 |
| `file_edits_json` | TEXT | 与消息关联的文件编辑数组 |
| `media_json` | TEXT | 关联媒体数组 |
| `created_at` | TEXT | 创建时间 |
| `updated_at` | TEXT | 更新时间 |

说明：

- 这是“前端消费投影”，不是事实源。
- `tool_events_json` 和 `file_edits_json` 当前可以先以 JSON 数组形式存储，后续如果需要更强查询能力再拆表。
- 线程加载优先读这个表，而不是实时重放所有事件。

### 6.5 `session_snapshot` 快照表

用途：

- 快速恢复大型会话
- 避免每次都从第一个事件重放
- 给导出、诊断、迁移提供稳定切面

建议字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | TEXT | 快照主键 |
| `session_id` | TEXT | 所属会话 |
| `turn_id` | TEXT | 生成该快照时对应的最新回合 |
| `snapshot_type` | TEXT | `thread/runtime/full` |
| `base_event_seq` | INTEGER | 快照覆盖到的最后事件序号 |
| `payload_json` | TEXT | 快照内容 |
| `created_at` | TEXT | 创建时间 |

建议快照内容至少包含：

- 当前线程消息数组
- 当前会话摘要
- 当前运行状态
- 工作区范围
- 最后已应用事件位置

### 6.6 `session_asset` 资产表

用途：

- 保存媒体附件、文件编辑 diff、导出文件等外部对象的索引

建议字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | TEXT | 资产主键 |
| `session_id` | TEXT | 所属会话 |
| `turn_id` | TEXT NULL | 所属回合 |
| `message_id` | TEXT NULL | 所属消息 |
| `asset_type` | TEXT | `media/diff/export/log` |
| `storage_path` | TEXT | 文件相对路径 |
| `mime_type` | TEXT NULL | 媒体类型 |
| `size_bytes` | INTEGER | 文件大小 |
| `checksum` | TEXT NULL | 校验值 |
| `meta_json` | TEXT | 附加元数据 |
| `created_at` | TEXT | 创建时间 |

说明：

- 二进制或大文本文件不要直接塞进事件表。
- 事件表只保存引用关系与必要描述。

---

## 7. 事件模型设计

### 7.1 事件分类

建议把事件分成以下几类：

1. 会话生命周期事件
2. 回合生命周期事件
3. 消息内容事件
4. 推理过程事件
5. 工具调用事件
6. 文件编辑事件
7. 运行异常事件
8. 工作区/上下文变更事件

### 7.2 建议事件类型

第一版建议至少定义以下事件类型：

| 事件类型 | 说明 |
| --- | --- |
| `session_created` | 创建会话 |
| `session_title_updated` | 修改会话标题 |
| `workspace_scope_updated` | 更新工作区范围 |
| `turn_started` | 一轮执行开始 |
| `user_message_created` | 用户消息落盘 |
| `assistant_message_started` | 助手消息开始生成 |
| `assistant_delta_appended` | 助手正文分块追加 |
| `assistant_reasoning_appended` | reasoning 分块追加 |
| `tool_event_recorded` | 工具调用事件 |
| `file_edit_recorded` | 文件编辑事件 |
| `media_asset_linked` | 关联媒体资产 |
| `goal_status_updated` | 运行状态更新 |
| `stream_finished` | 当前流式输出结束 |
| `turn_completed` | 一轮执行完成 |
| `turn_failed` | 一轮执行失败 |
| `turn_interrupted` | 一轮执行中断 |
| `session_deleted` | 会话软删除 |

### 7.3 关于 delta 事件的存储粒度

如果把每个 token 都单独写入数据库，会带来明显的写放大和存储膨胀问题。因此推荐策略是：

1. 内存中先积累增量片段。
2. 按“时间窗口”或“字符长度阈值”做分块落盘。
3. 回合结束时做一次最终归并更新消息投影。

建议默认策略：

- 每 `250ms` 或每 `500~1000` 字符 flush 一次
- reasoning 与正文分开分块
- 对最终 `session_message.content` 始终保存完整聚合结果

这样可以兼顾：

- 中途崩溃时尽量保留已生成内容
- 不把数据库写入压力推得过高

---

## 8. 读写流程设计

### 8.1 创建会话流程

1. 生成 `session_id`
2. 写入 `session`
3. 写入 `session_created` 事件
4. 初始化空快照或延迟到首轮消息后生成

返回值仍可保持当前接口形状：

```json
{
  "session": {
    "key": "xxx",
    "title": "New chat",
    "created_at": "...",
    "updated_at": "...",
    "preview": "",
    "run_started_at": null,
    "workspace_scope": null
  }
}
```

### 8.2 提交用户消息流程

当调用 `POST /api/sessions/{session_key}/messages` 时，建议按以下顺序处理：

1. 校验会话存在且未删除
2. 创建 `turn`
3. 写入 `user_message_created` 事件
4. 写入或更新对应 `session_message`
5. 写入 `turn_started` 事件
6. 更新 `session.status = running`
7. 执行 Agent / Gateway 逻辑
8. 执行期间持续写入中间事件
9. 回合结束后写入 `turn_completed` 或 `turn_failed`
10. 更新消息投影、会话摘要、运行状态
11. 必要时生成新快照

### 8.3 助手输出流程

当助手开始输出时：

1. 先创建一条空的 assistant 投影消息
2. 写入 `assistant_message_started`
3. 流式过程中持续写 `assistant_delta_appended`
4. reasoning 单独写 `assistant_reasoning_appended`
5. 工具调用写 `tool_event_recorded`
6. 文件修改写 `file_edit_recorded`
7. 输出结束后更新该消息为非 streaming

这样前端线程和底层日志是同步可解释的。

### 8.4 加载会话列表流程

`GET /api/sessions` 不应该扫描全量事件，而应该直接读取 `session` 表中的摘要字段。

排序依据：

- `updated_at DESC`

过滤建议：

- 默认过滤 `deleted_at IS NULL`

这样可以确保会话列表读取始终很快。

### 8.5 加载线程流程

`GET /api/sessions/{session_key}/webui-thread` 应从 `session_message` 投影表读取，而不是每次重建。

返回内容建议保持与当前前端兼容：

- `key`
- `messages`
- `workspace_scope`
- `has_pending_tool_calls`
- `page`

其中：

- `messages` 来源于 `session_message`
- `has_pending_tool_calls` 来源于 `session.status` 或最近 `running turn`

### 8.6 加载完整执行记录流程

为后续调试、审计、恢复，建议新增接口：

- `GET /api/sessions/{session_key}/turns`
- `GET /api/sessions/{session_key}/turns/{turn_id}/events`

这两个接口不是第一步必须立刻暴露给前端，但底层存储必须从一开始就能支持。

### 8.7 服务重启后的恢复流程

服务启动时应执行一次恢复扫描：

1. 查找 `session_turn.status = running`
2. 判断 `heartbeat_at` 是否超出阈值
3. 对超时 turn 标记为 `interrupted`
4. 写入 `turn_interrupted` 事件
5. 更新 `session.status`

这样用户重新打开历史会话时，可以明确看到“上次任务中断了”，而不是假装它已经成功完成。

---

## 9. 一致性设计

### 9.1 事实源与投影分离

必须明确：

- `session_event` 是事实源
- `session_message` 和 `session` 是投影
- `session_snapshot` 是恢复优化手段

如果投影损坏，理论上应该可以通过事件日志重建。

### 9.2 事务边界

建议事务边界如下：

1. 创建 turn + 用户消息事件 + 会话状态更新：一个事务
2. 每次中间事件 flush：一个小事务
3. 回合结束写入完成状态 + 投影最终收敛 + 快照更新：一个事务

这样做的目的不是追求“单事务包住整轮执行”，而是兼顾：

- 中途过程可持久化
- 数据库锁时间可控
- 崩溃后可恢复

### 9.3 幂等性

由于未来可能引入重试机制，关键写操作应具备幂等支持：

- `turn_id` 必须唯一
- `event_seq` 在同一 turn 内唯一
- 对同一 `turn_id + event_seq` 的重复写入应拒绝或忽略

### 9.4 乐观并发

虽然当前是单机单用户，但仍建议为 `session.version` 保留乐观并发控制能力，原因是：

- 前端可能快速连续提交
- 后续可能增加后台任务、恢复任务、批量导入
- 预留版本号远比未来补加更容易

---

## 10. 快照与重建策略

### 10.1 为什么需要快照

如果一个会话有几千条事件，每次都从头重放，会有两个问题：

- 线程加载变慢
- 恢复任务成本变高

所以需要快照。

### 10.2 快照触发时机

推荐触发条件：

1. 每轮任务结束后生成轻量线程快照
2. 每累计 `N` 个事件后生成完整快照
3. 在服务空闲时做后台压缩

建议初始值：

- 每轮结束写 1 个 `thread` 快照
- 每 100 个事件或每 10 轮额外写 1 个 `full` 快照

### 10.3 重建策略

重建线程时优先级如下：

1. 读取最新快照
2. 从快照的 `base_event_seq` 之后继续应用事件
3. 生成新的线程结果

如果没有快照：

1. 直接从事件表顺序重放
2. 生成投影
3. 回写快照

---

## 11. 性能与容量设计

### 11.1 读取性能目标

建议的第一版目标：

- 会话列表读取：`< 100ms`
- 单个线程加载：`< 200ms`
- 单次事件 flush：平均 `10~30ms` 量级
- 1000 个会话、10 万级事件下仍可稳定工作

### 11.2 索引设计

建议至少建立以下索引：

- `session(updated_at DESC)`
- `session_turn(session_id, turn_seq DESC)`
- `session_turn(session_id, status)`
- `session_event(session_id, turn_id, event_seq)`
- `session_event(session_id, created_at)`
- `session_message(session_id, message_seq)`
- `session_snapshot(session_id, created_at DESC)`
- `session_asset(session_id, created_at DESC)`

### 11.3 压缩策略

事件日志会持续增长，因此需要压缩策略：

1. 对已完成 turn 的高频 delta 事件做归并标记
2. 保留原始 turn 边界和关键里程碑事件
3. 大附件单独放文件系统，不放数据库
4. 对导出或长期历史支持归档

注意：

- 压缩不应破坏恢复能力
- 压缩后的事件仍然需要能解释线程最终状态

---

## 12. 与现有代码的对接方案

### 12.1 当前代码的关键挂点

现有代码中，主要改造点有三个：

1. `src/router/sessions_api.py`
2. `src/router/realtime.py`
3. `src/router/stream_aggregator.py` 对应的数据来源

### 12.2 建议新增的模块边界

建议新增如下模块：

```text
src/store/
  __init__.py
  base.py                # SessionStore 抽象接口
  models.py              # 存储层对象定义
  sqlite_store.py        # SQLite 实现
  projector.py           # 事件 -> 消息投影
  snapshotter.py         # 快照读写
  recovery.py            # 启动恢复逻辑
  migrations.py          # 数据库初始化与迁移
```

### 12.3 `SessionRepository` 的演进方式

不建议直接删掉 `SessionRepository`，而建议把它逐步演进成“应用层门面”：

- 第一阶段：`SessionRepository` 内部改为委托 `SessionStore`
- 第二阶段：把现有内存逻辑替换为 SQLite 实现
- 第三阶段：把事件追加、快照和恢复逻辑逐步下沉到 `store` 层

这样可以保持现有 router 基本不动，降低改造风险。

### 12.4 `HttpMessageGateway` 的改造重点

当前 `HttpMessageGateway.submit_message()` 是一次性拿到整批事件后再返回。

后续改造时建议：

1. 在收到用户消息时立即创建 turn 并落盘
2. 执行器每产生一类事件就交给 `SessionStore.append_event(...)`
3. 回合结束时统一 finalize

即使仍然沿用当前 HTTP 批量返回模式，底层也应该先写持久化，再返回响应。

---

## 13. API 兼容与扩展建议

### 13.1 保持现有接口不变

第一阶段建议保留以下接口形状不变：

- `GET /api/sessions`
- `POST /api/sessions`
- `GET /api/sessions/{session_key}/webui-thread`
- `DELETE /api/sessions/{session_key}`
- `POST /api/sessions/{session_key}/messages`

这样前端几乎不需要立即改动。

### 13.2 后续新增接口

建议逐步新增以下接口：

- `GET /api/sessions/{session_key}/turns`
- `GET /api/sessions/{session_key}/turns/{turn_id}`
- `GET /api/sessions/{session_key}/turns/{turn_id}/events`
- `POST /api/sessions/{session_key}/export`
- `POST /api/sessions/import`
- `POST /api/sessions/{session_key}/rebuild`

这些接口可以为后续功能提供支撑：

- 查看历史执行细节
- 导出完整任务轨迹
- 从事件日志重建线程
- 导入外部会话包

---

## 14. 删除、归档与导出策略

### 14.1 删除策略

第一版建议采用软删除：

- `session.deleted_at` 标记删除
- 会话列表默认不展示已删除项
- 真正清理由单独的 purge 操作完成

原因：

- 用户误删后仍有恢复空间
- 方便导出和诊断
- 不会立即破坏事件引用关系

### 14.2 归档策略

对于长期不活跃的大型会话，支持归档：

- 会话主记录保留
- 投影保留摘要
- 详细事件可压缩或转归档包

### 14.3 导出格式

建议导出为一个目录或压缩包，内容包括：

- `session.json`
- `turns.json`
- `events.ndjson`
- `messages.json`
- `assets/`

这样后续无论是导入、调试还是分享，都有稳定格式。

---

## 15. 故障场景设计

### 15.1 服务在执行中崩溃

预期行为：

1. 已 flush 的事件仍然存在
2. 未完成 turn 在下次启动时被标记为 `interrupted`
3. 线程中可看到不完整 assistant 输出
4. 用户可以重新发起任务

### 15.2 投影损坏但事件仍在

预期行为：

1. 使用事件日志 + 快照重建 `session_message`
2. 重建后更新 `session.latest_snapshot_id`
3. 不影响原始事实数据

### 15.3 资产文件缺失

预期行为：

1. 线程保留引用信息
2. 前端可显示“资产不可用”
3. 不阻断会话本身的读取

---

## 16. 分阶段落地方案

### 阶段一：落地最小可用持久化

目标：

- 用 SQLite 替代内存会话仓库
- 落盘 `session`、`session_turn`、`session_message`
- 保持当前接口不变

这一阶段先解决“历史会话不丢失”。

### 阶段二：补全事件日志

目标：

- 引入 `session_event`
- 把 `HttpMessageGateway` 的执行事件写入事实层
- 支持完整过程记录

这一阶段解决“执行过程可追踪”。

### 阶段三：补全快照与恢复

目标：

- 引入 `session_snapshot`
- 支持启动恢复、重建线程、处理中断回合

这一阶段解决“可恢复、可重建”。

### 阶段四：补全附件、导出、归档

目标：

- 引入 `session_asset`
- 支持导入导出
- 支持归档与清理

这一阶段解决“长生命周期运维能力”。

---

## 17. 关键设计决策总结

### 决策一：使用事件日志 + 投影，而不是只存消息

原因：

- 需要保存执行过程
- 需要可审计、可恢复、可重建

### 决策二：第一版默认使用 SQLite

原因：

- 项目当前是本地单机场景
- 部署简单、事务能力足够

### 决策三：事实层和展示层分离

原因：

- 前端要快，底层要准
- 投影损坏后必须还能重建

### 决策四：保留现有路由接口，先替换仓库实现

原因：

- 风险最小
- 便于分阶段交付

---

## 18. 最终建议

对于当前项目，这个会话存储系统最合适的落地路径不是“一步做成复杂平台”，而是：

1. 先把 `SessionRepository` 从内存仓库升级为基于 `SessionStore` 抽象的持久化仓库。
2. 第一版用 SQLite 承载会话主表、回合表、消息投影表。
3. 第二版引入追加式事件日志，正式把“执行过程”纳入事实层。
4. 第三版补上快照、恢复、导出、归档能力。

如果后续严格按照这个设计推进，那么系统将具备以下能力：

- 用户可以稳定查看历史会话
- 用户可以回看任务执行过程
- 系统可以识别和标记异常中断
- 后端存储可以支撑更复杂的 Agent 与工具链
- 当前前端接口不需要大规模推翻重写

这会让“会话”从一个短生命周期的内存对象，升级成一个真正可管理、可追踪、可恢复的任务资产。
