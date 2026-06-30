# AtomCode2API

将 AtomCode AI 编程助手包装成 OpenAI 兼容 API，可直接在 **Cursor**、**Trae** 等 IDE 中使用。

---

## 做这个项目的起因

最近发现 AtomCode 的 **CodingPlan** 挺香的 —— 每 5 小时就有 **800 次 deepseek-v4-flash** 的调用额度，而且速度很快。关键是现在可以免费薅这个羊毛，对于日常写码查问题来说完全够用。

#### [点击这里注册](https://atomcode.atomgit.com/invite/UVMWDFM7)

---

## 快速开始

**无需 Python 环境：** 直接运行 `atomcode2api.exe` 即可 [下载地址](https://github.com/Solongbus/AtomCode2API/releases)

或

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行启动脚本
start_atomcode2api.bat
```

---

## 在 IDE 中配置

| 配置项 | 值 |
|--------|-----|
| **Base URL** | `http://127.0.0.1:8123/v1` |
| **API Key** | `atomcode2api-local-dev-key`（由 `start_atomcode2api.bat` 设定；留空会返回 401） |
| **Model** | `deepseek-v4-flash` 或 `Qwen/Qwen3-VL-8B-Instruct`（二选一，详见 `GET /v1/models`） |

> API Key 的值取自 `start_atomcode2api.bat` 中的 `ATOMCODE_API_KEY` 环境变量，如需更换，同步修改启动脚本和 IDE 配置即可。

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/chat/completions` | OpenAI 兼容聊天（支持流式 SSE） |
| `GET` | `/v1/models` | 列出可用模型 |
| `GET` | `/health` | 健康检查 |

---

## 配置项

通过 `ATOMCODE_` 前缀的环境变量配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ATOMCODE_API_KEY` | `""` | API 鉴权密钥（空则不鉴权） |
| `ATOMCODE_HOST` | `0.0.0.0` | 监听地址 |
| `ATOMCODE_PORT` | `8123` | 监听端口 |
| `ATOMCODE_MODE` | `cli` | 运行模式：`daemon` 或 `cli` |
| `ATOMCODE_DEFAULT_WORKSPACE` | `""` | 默认工作目录 |
| `ATOMCODE_TIMEOUT` | `600` | 任务超时（秒） |

---

## 请求示例

```json
POST /v1/chat/completions
{
  "model": "deepseek-v4-flash",
  "messages": [
    { "role": "user", "content": "创建一个登录页面" }
  ],
  "stream": true
}
```

---
