# 聊天系统项目

## 项目简介
这是一个基于FastAPI的聊天系统，包含前后端代码。系统支持顾客(customer)和商家(merchant)两种角色，实现了聊天功能和商品管理功能。

## 项目结构
```
tuijianxiton/
├── backend/              # 后端代码
│   ├── app/
│   │   ├── api/          # API路由
│   │   ├── models/       # 数据库模型
│   │   ├── schemas/      # 数据验证模式
│   │   ├── services/     # 业务逻辑
│   │   ├── utils/        # 工具函数
│   │   └── database.py   # 数据库配置
│   ├── main.py           # 主应用文件
│   └── requirements.txt  # 依赖文件
├── frontend/             # 前端代码
│   ├── customer/         # 顾客端
│   └── merchant/         # 商家端
├── docs/                 # 文档
│   ├── api_docs.md       # API文档
│   └── database_schema.md # 数据库 schema
└── README.md             # 项目说明
```

## 技术栈
- **后端**：FastAPI, SQLAlchemy, PostgreSQL, JWT认证
- **前端**：HTML, CSS, JavaScript
- **数据库**：PostgreSQL

## 环境要求
1. Python 3.8+
2. PostgreSQL 10+
3. 现代浏览器（用于前端访问）

## 数据库准备

### 1. 启动PostgreSQL服务
确保PostgreSQL数据库服务已启动。

### 2. 创建数据库和用户
使用以下命令创建数据库和用户（可通过pgAdmin或psql命令行执行）：

```sql
-- 创建用户
CREATE USER myuser WITH PASSWORD '123456';

-- 创建数据库
CREATE DATABASE mydb OWNER myuser;

-- 授予权限
GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;
```

### 3. 数据库表初始化
数据库表会在应用启动时自动创建，无需手动执行SQL脚本。

## 依赖安装

1. 进入后端目录：
```bash
cd backend
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 启动步骤

### 1. 启动后端服务
在backend目录下执行：
```bash
python main.py
```
或使用uvicorn直接启动：
```bash
uvicorn main:app --reload
```

后端服务默认运行在 `http://127.0.0.1:8000`

### 2. 访问前端页面
- **顾客端**：打开浏览器访问 `frontend/customer/index.html`
- **商家端**：打开浏览器访问 `frontend/merchant/index.html`

## API文档

FastAPI自动生成了交互式API文档：
- **Swagger UI**：访问 `http://127.0.0.1:8000/docs`
- **ReDoc**：访问 `http://127.0.0.1:8000/redoc`

## 系统功能

### 后端API
- **认证**：用户注册、登录、获取令牌
- **聊天**：创建聊天室、发送消息、获取消息历史
- **商品**：商家发布商品、顾客查看商品

### 前端功能
- **顾客端**：查看商品、与商家聊天
- **商家端**：管理商品、与顾客聊天

## 注意事项

1. **数据库连接配置**：
   数据库连接信息在 `backend/app/database.py` 中配置，默认值为：
   ```python
   DATABASE_URL = "postgresql://myuser:123456@127.0.0.1/mydb"
   ```
   如果你的数据库配置不同，请修改此值。

2. **CORS配置**：
   为了方便开发，CORS已设置为允许所有来源。在生产环境中，应设置具体的前端域名。

3. **安全提示**：
   - 密码使用bcrypt加密存储
   - 使用JWT进行身份验证
   - 生产环境中应使用HTTPS

## 故障排除

### 常见问题

1. **数据库连接失败**：
   - 检查PostgreSQL服务是否运行
   - 确认数据库和用户已正确创建
   - 验证数据库连接字符串是否正确

2. **依赖安装失败**：
   - 确保使用了正确的Python版本
   - 尝试使用pip的--upgrade选项

3. **前端无法连接后端**：
   - 确认后端服务已启动
   - 检查浏览器控制台是否有网络错误
   - 验证CORS配置是否正确

## 开发指南

### 添加新功能
1. 在 `app/models/` 中定义数据库模型
2. 在 `app/schemas/` 中定义数据验证模式
3. 在 `app/services/` 中实现业务逻辑
4. 在 `app/api/` 中添加API路由
5. 运行 `python main.py` 启动服务

### 测试API
使用Swagger UI（`/docs`）或Postman等工具测试API端点。
