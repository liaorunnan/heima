// API基础URL
const API_BASE_URL = 'http://localhost:8000';

// DOM元素
const authContainer = document.getElementById('auth-container');
const merchantContainer = document.getElementById('merchant-container');
const loginTab = document.getElementById('login-tab');
const registerTab = document.getElementById('register-tab');
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const loginBtn = document.getElementById('login-btn');
const registerBtn = document.getElementById('register-btn');
const authMessage = document.getElementById('auth-message');
const logoutBtn = document.getElementById('logout-btn');
const navChat = document.getElementById('nav-chat');
const navProducts = document.getElementById('nav-products');
const chatManagement = document.getElementById('chat-management');
const productManagement = document.getElementById('product-management');
const createChatBtn = document.getElementById('create-chat-btn');
const chatRoomsList = document.getElementById('chat-rooms-list');
const chatEmpty = document.getElementById('chat-empty');
const chatWindow = document.getElementById('chat-window');
const chatRoomTitle = document.getElementById('chat-room-title');
const messagesContainer = document.getElementById('messages-container');
const messageInput = document.getElementById('message-input');
const selectProductBtn = document.getElementById('select-product-btn');
const sendMessageBtn = document.getElementById('send-message-btn');
const sendProductBtn = document.getElementById('send-product-btn');
const productsList = document.getElementById('products-list');
const addProductBtn = document.getElementById('add-product-btn');
const productSelectModal = document.getElementById('product-select-modal');
const selectableProducts = document.getElementById('selectable-products');
const cancelSelectProduct = document.getElementById('cancel-select-product');
const productModal = document.getElementById('product-modal');
const productDetails = document.getElementById('product-details');
const addProductModal = document.getElementById('add-product-modal');
const createChatModal = document.getElementById('create-chat-modal');
const customerUsername = document.getElementById('customer-username');
const cancelCreateChat = document.getElementById('cancel-create-chat');
const confirmCreateChat = document.getElementById('confirm-create-chat');
const modalTitle = document.getElementById('modal-title');
const productForm = document.getElementById('product-form');
const productId = document.getElementById('product-id');
const productName = document.getElementById('product-name');
const productDescription = document.getElementById('product-description');
const productPrice = document.getElementById('product-price');
const cancelProduct = document.getElementById('cancel-product');
const saveProduct = document.getElementById('save-product');

// 当前状态
let currentUser = null;
let currentChatRoom = null;
let chatRooms = [];
let products = [];
let selectedProductId = null;
let messagePollingInterval = null;

// 认证相关函数
function showAuthMessage(message, isError = false) {
    authMessage.textContent = message;
    authMessage.className = isError ? 'error' : 'success';
    setTimeout(() => {
        authMessage.textContent = '';
        authMessage.className = '';
    }, 3000);
}

function switchAuthTab(tab) {
    if (tab === 'login') {
        loginTab.classList.add('active');
        registerTab.classList.remove('active');
        loginForm.classList.add('active');
        registerForm.classList.remove('active');
    } else {
        registerTab.classList.add('active');
        loginTab.classList.remove('active');
        registerForm.classList.add('active');
        loginForm.classList.remove('active');
    }
}

async function login() {
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;

    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                username: username,
                password: password,
                expected_role: 'merchant',
            }),
        });

        if (!response.ok) {
            throw new Error('登录失败');
        }

        const data = await response.json();
        localStorage.setItem('token', data.access_token);
        localStorage.setItem('username', username);
        await loadUserInfo();
    } catch (error) {
        showAuthMessage('登录失败，请检查用户名和密码，或您不是商家账号', true);
        console.error('登录错误:', error);
    }
}

async function register() {
    const username = document.getElementById('register-username').value;
    const password = document.getElementById('register-password').value;

    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                username: username,
                password: password,
                role: 'merchant',
            }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '注册失败');
        }

        showAuthMessage('注册成功，请登录');
        switchAuthTab('login');
    } catch (error) {
        showAuthMessage(error.message, true);
        console.error('注册错误:', error);
    }
}

// 解析JWT token
function parseJwt(token) {
    try {
        const base64Url = token.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));
        return JSON.parse(jsonPayload);
    } catch (error) {
        console.error('解析token错误:', error);
        return null;
    }
}

async function loadUserInfo() {
    try {
        const token = localStorage.getItem('token');
        if (!token) {
            return;
        }

        // 解析JWT token获取用户信息
        const tokenData = parseJwt(token);
        if (tokenData) {
            // 从token中获取用户ID（整数）和角色
            currentUser = {
                id: parseInt(tokenData.sub),
                role: tokenData.role,
                username: tokenData.username
            };
        } else {
            // 如果无法解析token，使用用户名作为ID（后备方案）
            const username = localStorage.getItem('username') || 'merchant';
            currentUser = { id: username, role: 'merchant' };
        }

        // 验证token是否有效
        const productsResponse = await fetch(`${API_BASE_URL}/api/products`, {
            headers: {
                'Authorization': `Bearer ${token}`,
            },
        });

        if (!productsResponse.ok) {
            localStorage.removeItem('token');
            localStorage.removeItem('username');
            return;
        }

        showMerchantInterface();
        await loadChatRooms();
        await loadProducts();
        // 启动消息实时更新
        startMessagePolling();
    } catch (error) {
        console.error('加载用户信息错误:', error);
        localStorage.removeItem('token');
        localStorage.removeItem('username');
    }
}

function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    currentUser = null;
    currentChatRoom = null;
    chatRooms = [];
    products = [];
    selectedProductId = null;
    stopMessagePolling();
    showAuthInterface();
}

function showAuthInterface() {
    authContainer.classList.remove('hidden');
    merchantContainer.classList.add('hidden');
}

function showMerchantInterface() {
    authContainer.classList.add('hidden');
    merchantContainer.classList.remove('hidden');
}

function switchContentSection(section) {
    if (section === 'chat') {
        navChat.classList.add('active');
        navProducts.classList.remove('active');
        chatManagement.classList.add('active');
        productManagement.classList.remove('active');
    } else {
        navProducts.classList.add('active');
        navChat.classList.remove('active');
        productManagement.classList.add('active');
        chatManagement.classList.remove('active');
    }
}

// 聊天相关函数
async function loadChatRooms() {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`${API_BASE_URL}/api/chat/rooms`, {
            headers: {
                'Authorization': `Bearer ${token}`,
            },
        });

        if (!response.ok) {
            throw new Error('获取聊天列表失败');
        }

        chatRooms = await response.json();
        renderChatRooms();
    } catch (error) {
        console.error('加载聊天列表错误:', error);
    }
}

function renderChatRooms() {
    chatRoomsList.innerHTML = '';

    if (chatRooms.length === 0) {
        chatRoomsList.innerHTML = '<p class="empty-state">暂无聊天记录</p>';
        return;
    }

    chatRooms.forEach(room => {
        const roomItem = document.createElement('div');
        roomItem.className = 'chat-room-item';
        roomItem.dataset.roomId = room.id;

        roomItem.innerHTML = `
            <div class="chat-room-name">客户 ${room.customer_id}</div>
            <div class="chat-room-last-message">${room.last_message || '无消息'}</div>
            <div class="chat-room-time">${room.last_message_time ? new Date(room.last_message_time).toLocaleString() : ''}</div>
        `;

        roomItem.addEventListener('click', () => {
            selectChatRoom(room.id);
        });

        chatRoomsList.appendChild(roomItem);
    });
}

async function selectChatRoom(roomId) {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`${API_BASE_URL}/api/chat/rooms/${roomId}`, {
            headers: {
                'Authorization': `Bearer ${token}`,
            },
        });

        if (!response.ok) {
            throw new Error('获取聊天室详情失败');
        }

        currentChatRoom = await response.json();
        showChatWindow();
        renderMessages();

        // 更新聊天室列表的选中状态
        document.querySelectorAll('.chat-room-item').forEach(item => {
            item.classList.remove('active');
            if (parseInt(item.dataset.roomId) === roomId) {
                item.classList.add('active');
            }
        });
    } catch (error) {
        console.error('选择聊天室错误:', error);
    }
}

function showChatWindow() {
    chatEmpty.classList.add('hidden');
    chatWindow.classList.remove('hidden');
    chatRoomTitle.textContent = `客户 ${currentChatRoom.customer_id}`;
}

function renderMessages() {
    messagesContainer.innerHTML = '';

    if (!currentChatRoom.messages || currentChatRoom.messages.length === 0) {
        messagesContainer.innerHTML = '<p class="empty-state">暂无消息</p>';
        return;
    }

    currentChatRoom.messages.forEach(message => {
        addMessageToDOM(message);
    });

    // 滚动到底部
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addMessageToDOM(message) {
    const messageElement = document.createElement('div');
    // 根据发送者角色区分消息位置和样式
    // 添加调试信息
    console.log('=== 消息调试信息 ===');
    console.log('Message sender_id:', message.sender_id);
    console.log('Message sender_role:', message.sender_role);
    console.log('Current user id:', currentUser.id);
    console.log('Current user role:', currentUser.role);
    // 根据发送者角色判断：商家发的消息在右边，客户发的消息在左边
    const isSentByMerchant = message.sender_role === 'merchant';
    console.log('Is sent by merchant:', isSentByMerchant);
    console.log('Message class:', isSentByMerchant ? 'message-sent' : 'message-received');
    console.log('===================');
    messageElement.className = `message ${isSentByMerchant ? 'message-sent' : 'message-received'}`;

    if (message.message_type === 'product' && message.product_id) {
        // 显示商品卡片
        messageElement.innerHTML = `
            <div class="product-card" onclick="showProductDetails(${message.product_id})">
                <div class="product-name">${message.content}</div>
                <div class="product-price">¥${parseFloat(message.product_price).toFixed(2)}</div>
            </div>
            <div class="message-time">${new Date(message.created_at).toLocaleString()}</div>
        `;
    } else {
        // 显示普通消息
        messageElement.innerHTML = `
            <div class="message-content">${message.content}</div>
            <div class="message-time">${new Date(message.created_at).toLocaleString()}</div>
        `;
    }

    messagesContainer.appendChild(messageElement);
    // 滚动到底部
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

async function sendMessage() {
    const content = messageInput.value.trim();
    if (!content || !currentChatRoom) {
        return;
    }

    // 立即清空输入框，提供更好的用户体验
    messageInput.value = '';
    
    try {
        const token = localStorage.getItem('token');
        const messageData = {
            content: content,
            message_type: 'text',
            product_id: null
        };

        const response = await fetch(`${API_BASE_URL}/api/chat/rooms/${currentChatRoom.id}/messages`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
            },
            body: JSON.stringify(messageData),
        });

        if (!response.ok) {
            throw new Error('发送消息失败');
        }

        const newMessage = await response.json();
        
        // 立即添加新消息到界面，避免卡顿
        if (!currentChatRoom.messages) {
            currentChatRoom.messages = [];
        }
        currentChatRoom.messages.push(newMessage);
        
        // 只添加新消息到DOM，避免重新渲染所有消息
        addMessageToDOM(newMessage);
    } catch (error) {
        console.error('发送消息错误:', error);
        // 如果发送失败，恢复输入框内容
        messageInput.value = content;
    }
}

async function sendSelectedProduct() {
    if (!selectedProductId || !currentChatRoom) {
        alert('请选择一个商品');
        return;
    }

    try {
        const token = localStorage.getItem('token');
        const product = products.find(p => p.id === selectedProductId);
        const messageData = {
            content: product.name,
            message_type: 'product',
            product_id: selectedProductId
        };

        const response = await fetch(`${API_BASE_URL}/api/chat/rooms/${currentChatRoom.id}/messages`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
            },
            body: JSON.stringify(messageData),
        });

        if (!response.ok) {
            throw new Error('发送商品失败');
        }

        const newMessage = await response.json();
        productSelectModal.classList.add('hidden');
        selectedProductId = null;
        
        // 立即添加新消息到界面，避免卡顿
        if (!currentChatRoom.messages) {
            currentChatRoom.messages = [];
        }
        currentChatRoom.messages.push(newMessage);
        
        // 只添加新消息到DOM，避免重新渲染所有消息
        addMessageToDOM(newMessage);
    } catch (error) {
        console.error('发送商品错误:', error);
        alert('发送商品失败，请重试');
    }
}

async function createChatWithCustomer() {
    const username = customerUsername.value.trim();
    if (!username) {
        alert('请输入客户用户名');
        return;
    }

    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`${API_BASE_URL}/api/chat/rooms/by-username`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
            },
            body: JSON.stringify({
                target_username: username,
            }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '创建聊天失败');
        }

        const chatRoom = await response.json();
        createChatModal.classList.add('hidden');
        customerUsername.value = '';
        await loadChatRooms();
        await selectChatRoom(chatRoom.id);
    } catch (error) {
        console.error('创建聊天错误:', error);
        alert(error.message);
    }
}

// 商品相关函数
async function loadProducts() {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`${API_BASE_URL}/api/products`, {
            headers: {
                'Authorization': `Bearer ${token}`,
            },
        });

        if (!response.ok) {
            throw new Error('获取商品列表失败');
        }

        products = await response.json();
        renderProducts();
        renderSelectableProducts();
    } catch (error) {
        console.error('加载商品列表错误:', error);
    }
}

function renderProducts() {
    productsList.innerHTML = '';

    if (products.length === 0) {
        productsList.innerHTML = '<p class="empty-state">暂无商品，请添加商品</p>';
        return;
    }

    products.forEach(product => {
        const productItem = document.createElement('div');
        productItem.className = 'product-item';

        productItem.innerHTML = `
            <h3>${product.name}</h3>
            <p>${product.description}</p>
            <div class="product-price">¥${product.price}</div>
            <div class="product-actions">
                <button class="edit-product-btn" onclick="editProduct(${product.id})">编辑</button>
                <button class="delete-product-btn" onclick="deleteProduct(${product.id})">删除</button>
            </div>
        `;

        productsList.appendChild(productItem);
    });
}

function renderSelectableProducts() {
    selectableProducts.innerHTML = '';

    if (products.length === 0) {
        selectableProducts.innerHTML = '<p class="empty-state">暂无商品，请先添加商品</p>';
        return;
    }

    products.forEach(product => {
        const productItem = document.createElement('div');
        productItem.className = 'selectable-product-item';
        productItem.dataset.productId = product.id;

        productItem.innerHTML = `
            <div class="product-name">${product.name}</div>
            <div class="product-price">¥${product.price}</div>
        `;

        productItem.addEventListener('click', () => {
            selectedProductId = product.id;
            document.querySelectorAll('.selectable-product-item').forEach(item => {
                item.classList.remove('selected');
            });
            productItem.classList.add('selected');
        });

        selectableProducts.appendChild(productItem);
    });
}

async function showProductDetails(productId) {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`${API_BASE_URL}/api/products/${productId}`, {
            headers: {
                'Authorization': `Bearer ${token}`,
            },
        });

        if (!response.ok) {
            throw new Error('获取商品详情失败');
        }

        const product = await response.json();
        productDetails.innerHTML = `
            <h3>${product.name}</h3>
            <p><strong>价格:</strong> ¥${product.price}</p>
            <p><strong>描述:</strong> ${product.description}</p>
        `;
        productModal.classList.remove('hidden');
    } catch (error) {
        console.error('获取商品详情错误:', error);
    }
}

function openAddProductModal() {
    modalTitle.textContent = '添加商品';
    productId.value = '';
    productName.value = '';
    productDescription.value = '';
    productPrice.value = '';
    addProductModal.classList.remove('hidden');
}

function editProduct(productId) {
    const product = products.find(p => p.id === productId);
    if (product) {
        modalTitle.textContent = '编辑商品';
        productId.value = product.id;
        productName.value = product.name;
        productDescription.value = product.description;
        productPrice.value = product.price;
        addProductModal.classList.remove('hidden');
    }
}

async function saveProductData(e) {
    e.preventDefault();

    const id = productId.value;
    const name = productName.value;
    const description = productDescription.value;
    const price = productPrice.value;

    try {
        const token = localStorage.getItem('token');
        let response;

        if (id) {
            // 更新商品
            response = await fetch(`${API_BASE_URL}/api/products/${id}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`,
                },
                body: JSON.stringify({
                    name: name,
                    description: description,
                    price: price
                }),
            });
        } else {
            // 添加商品
            response = await fetch(`${API_BASE_URL}/api/products`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`,
                },
                body: JSON.stringify({
                    name: name,
                    description: description,
                    price: price
                }),
            });
        }

        if (!response.ok) {
            throw new Error('保存商品失败');
        }

        addProductModal.classList.add('hidden');
        await loadProducts();
    } catch (error) {
        console.error('保存商品错误:', error);
    }
}

async function deleteProduct(productId) {
    if (!confirm('确定要删除这个商品吗？')) {
        return;
    }

    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`${API_BASE_URL}/api/products/${productId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`,
            },
        });

        if (!response.ok) {
            throw new Error('删除商品失败');
        }

        await loadProducts();
    } catch (error) {
        console.error('删除商品错误:', error);
    }
}

// 事件监听器
loginTab.addEventListener('click', () => switchAuthTab('login'));
registerTab.addEventListener('click', () => switchAuthTab('register'));
loginBtn.addEventListener('click', login);
registerBtn.addEventListener('click', register);
logoutBtn.addEventListener('click', logout);
navChat.addEventListener('click', () => switchContentSection('chat'));
navProducts.addEventListener('click', () => switchContentSection('products'));
createChatBtn.addEventListener('click', () => createChatModal.classList.remove('hidden'));
cancelCreateChat.addEventListener('click', () => createChatModal.classList.add('hidden'));
confirmCreateChat.addEventListener('click', createChatWithCustomer);
selectProductBtn.addEventListener('click', () => productSelectModal.classList.remove('hidden'));
cancelSelectProduct.addEventListener('click', () => productSelectModal.classList.add('hidden'));
sendProductBtn.addEventListener('click', sendSelectedProduct);
sendMessageBtn.addEventListener('click', sendMessage);
addProductBtn.addEventListener('click', openAddProductModal);
cancelProduct.addEventListener('click', () => addProductModal.classList.add('hidden'));
saveProduct.addEventListener('click', saveProductData);

messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// 关闭模态框
const closeButtons = document.querySelectorAll('.close');
closeButtons.forEach(button => {
    button.addEventListener('click', () => {
        const modal = button.closest('.modal');
        modal.classList.add('hidden');
    });
});

// 点击模态框外部关闭
window.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.classList.add('hidden');
    }
});

// 消息实时更新机制
function startMessagePolling() {
    // 清除之前的轮询
    if (messagePollingInterval) {
        clearInterval(messagePollingInterval);
    }

    // 每3秒轮询一次消息
    messagePollingInterval = setInterval(async () => {
        if (currentChatRoom) {
            try {
                const token = localStorage.getItem('token');
                const response = await fetch(`${API_BASE_URL}/api/chat/rooms/${currentChatRoom.id}`, {
                    headers: {
                        'Authorization': `Bearer ${token}`,
                    },
                });

                if (response.ok) {
                    const updatedChatRoom = await response.json();
                    // 检查消息是否有更新
                    if (JSON.stringify(updatedChatRoom.messages) !== JSON.stringify(currentChatRoom.messages)) {
                        currentChatRoom = updatedChatRoom;
                        renderMessages();
                    }
                }
            } catch (error) {
                console.error('轮询消息错误:', error);
            }
        }
    }, 3000);
}

// 停止消息轮询
function stopMessagePolling() {
    if (messagePollingInterval) {
        clearInterval(messagePollingInterval);
        messagePollingInterval = null;
    }
}

// 初始化
async function init() {
    await loadUserInfo();
}

init();