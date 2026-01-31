# API 文档

## 1. 概述

本聊天系统提供了完整的 RESTful API，用于用户认证、聊天管理和商品管理。所有API端点都以 `/api` 为前缀，支持 JSON 格式的请求和响应。

## 2. 认证

### 2.1 认证方式

系统使用 JWT (JSON Web Token) 进行认证，获取令牌后，需要在后续请求的 `Authorization` 头中携带令牌。

### 2.2 认证端点

#### 2.2.1 注册

- **URL**: `/api/auth/register`
- **方法**: `POST`
- **请求体**:

```json
{
  "username": "string",
  "password": "string",
  "role": "customer|merchant"
}
```

- **响应**:

```json
{
  "id": 1,
  "username": "string",
  "role": "customer|merchant"
}
```

#### 2.2.2 登录

- **URL**: `/api/auth/login`
- **方法**: `POST`
- **请求体** (form-data):

```
username: string
password: string
```

- **响应**:

```json
{
  "access_token": "string",
  "token_type": "bearer"
}
```

## 3. 聊天

### 3.1 聊天室

#### 3.1.1 创建聊天室

- **URL**: `/api/chat/rooms`
- **方法**: `POST`
- **请求体**:

```json
{
  "customer_id": 1,
  "merchant_id": 2
}
```

- **响应**:

```json
{
  "id": 1,
  "customer_id": 1,
  "merchant_id": 2,
  "created_at": "2026-01-31T12:00:00Z"
}
```

#### 3.1.2 获取聊天室列表

- **URL**: `/api/chat/rooms`
- **方法**: `GET`
- **响应**:

```json
[
  {
    "id": 1,
    "customer_id": 1,
    "merchant_id": 2,
    "created_at": "2026-01-31T12:00:00Z",
    "last_message": "Hello",
    "last_message_time": "2026-01-31T12:01:00Z"
  }
]
```

#### 3.1.3 获取聊天室详情

- **URL**: `/api/chat/rooms/{room_id}`
- **方法**: `GET`
- **响应**:

```json
{
  "id": 1,
  "customer_id": 1,
  "merchant_id": 2,
  "created_at": "2026-01-31T12:00:00Z",
  "last_message": "Hello",
  "last_message_time": "2026-01-31T12:01:00Z",
  "messages": [
    {
      "id": 1,
      "chat_room_id": 1,
      "sender_id": 1,
      "content": "Hello",
      "message_type": "text",
      "product_id": null,
      "created_at": "2026-01-31T12:01:00Z"
    }
  ]
}
```

### 3.2 消息

#### 3.2.1 发送消息

- **URL**: `/api/chat/rooms/{room_id}/messages`
- **方法**: `POST`
- **请求体** (普通消息):

```json
{
  "content": "string",
  "message_type": "text",
  "product_id": null
}
```

- **请求体** (商品卡片):

```json
{
  "content": "商品: 商品名称",
  "message_type": "product",
  "product_id": 1
}
```

- **响应**:

```json
{
  "id": 1,
  "chat_room_id": 1,
  "sender_id": 1,
  "content": "string",
  "message_type": "text|product",
  "product_id": null|1,
  "created_at": "2026-01-31T12:01:00Z"
}
```

#### 3.2.2 获取消息列表

- **URL**: `/api/chat/rooms/{room_id}/messages`
- **方法**: `GET`
- **响应**:

```json
[
  {
    "id": 1,
    "chat_room_id": 1,
    "sender_id": 1,
    "content": "Hello",
    "message_type": "text",
    "product_id": null,
    "created_at": "2026-01-31T12:01:00Z"
  }
]
```

## 4. 商品

### 4.1 获取商品列表

- **URL**: `/api/products`
- **方法**: `GET`
- **响应**:

```json
[
  {
    "id": 1,
    "name": "商品名称",
    "description": "商品描述",
    "price": 99.99,
    "merchant_id": 2,
    "created_at": "2026-01-31T12:00:00Z"
  }
]
```

### 4.2 获取商品详情

- **URL**: `/api/products/{product_id}`
- **方法**: `GET`
- **响应**:

```json
{
  "id": 1,
  "name": "商品名称",
  "description": "商品描述",
  "price": 99.99,
  "merchant_id": 2,
  "created_at": "2026-01-31T12:00:00Z"
}
```

### 4.3 创建商品

- **URL**: `/api/products`
- **方法**: `POST`
- **请求体**:

```json
{
  "name": "商品名称",
  "description": "商品描述",
  "price": 99.99
}
```

- **响应**:

```json
{
  "id": 1,
  "name": "商品名称",
  "description": "商品描述",
  "price": 99.99,
  "merchant_id": 2,
  "created_at": "2026-01-31T12:00:00Z"
}
```

### 4.4 更新商品

- **URL**: `/api/products/{product_id}`
- **方法**: `PUT`
- **请求体**:

```json
{
  "name": "新商品名称",
  "description": "新商品描述",
  "price": 199.99
}
```

- **响应**:

```json
{
  "id": 1,
  "name": "新商品名称",
  "description": "新商品描述",
  "price": 199.99,
  "merchant_id": 2,
  "created_at": "2026-01-31T12:00:00Z"
}
```

### 4.5 删除商品

- **URL**: `/api/products/{product_id}`
- **方法**: `DELETE`
- **响应**: `204 No Content`

## 5. 错误处理

API 会返回标准的 HTTP 状态码来表示请求的结果，常见的错误状态码包括：

- `400 Bad Request`: 请求参数错误
- `401 Unauthorized`: 认证失败
- `403 Forbidden`: 权限不足
- `404 Not Found`: 资源不存在
- `500 Internal Server Error`: 服务器内部错误

错误响应格式：

```json
{
  "detail": "错误信息"
}
```

## 6. 示例请求

### 6.1 注册请求

```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "customer1",
    "password": "123456",
    "role": "customer"
  }'
```

### 6.2 登录请求

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=customer1&password=123456"
```

### 6.3 发送消息请求

```bash
curl -X POST http://localhost:8000/api/chat/rooms/1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "content": "Hello",
    "message_type": "text",
    "product_id": null
  }'
```

## 7. 注意事项

1. **令牌过期**：令牌有效期为 30 分钟，过期后需要重新登录获取新令牌
2. **权限控制**：商家只能管理自己的商品，用户只能访问自己的聊天室
3. **请求频率**：建议合理控制请求频率，避免过度请求导致服务器压力过大
4. **错误处理**：客户端应该妥善处理 API 返回的错误信息，为用户提供友好的提示

## 8. API 测试

系统使用 FastAPI 框架，提供了自动生成的 Swagger 文档，可以通过访问 `http://localhost:8000/docs` 查看和测试所有 API 端点。