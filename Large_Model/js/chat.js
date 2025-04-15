/**
 * chat.js - 聊天交互逻辑
 */

// 添加消息到聊天框
function addMessage(text, isUser = false) {
    const chatBox = document.getElementById('chat-box');
    
    // 创建消息元素
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.textContent = text;

    // 添加消息框淡入动画效果
    messageDiv.style.opacity = '0';
    chatBox.appendChild(messageDiv);
    // 触发动画
    setTimeout(() => {
        messageDiv.style.opacity = '1';
    }, 10);

    // 自动滚动到底部
    chatBox.scrollTo({
        top: chatBox.scrollHeight,
        behavior: 'smooth'
    });
}

// 显示正在输入状态的动画指示器
function showTypingIndicator() {
    const chatBox = document.getElementById('chat-box');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = '<div class="dot-flashing"></div>';
    chatBox.appendChild(typingDiv);
    chatBox.scrollTo({
        top: chatBox.scrollHeight,
        behavior: 'smooth'
    });
    return typingDiv;
}

// 发送消息处理
function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    
    if (!message) return;

    // 添加用户消息到聊天区域
    addMessage(message, true);
    input.value = '';

    // 显示机器人正在输入的指示器
    const typingIndicator = showTypingIndicator();

    // 模拟处理时间（例如调用后端 API 或 AI 模型处理）
    setTimeout(() => {
        // 移除正在输入提示
        typingIndicator.remove();
        
        // 生成倒序回复
        const reversedText = message.split('').reverse().join('');
        addMessage(reversedText);
    }, 800); // 模拟处理延迟 800ms
}

// 回车发送功能（避免与换行冲突，当Shift+Enter时允许换行）
document.getElementById('message-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// 输入框高度自适应
document.getElementById('message-input').addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight - 20 + 'px';
});

// 为发送按钮绑定点击事件
document.getElementById('send-btn').addEventListener('click', sendMessage);