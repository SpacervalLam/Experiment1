/**
 * main.js - 聊天交互逻辑
 */

// 添加消息到聊天框（支持 Markdown 格式）
function addMessage(text, isUser = false) {
    const chatBox = document.getElementById('chat-box');
    
    // 创建消息元素
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    // 解析 Markdown 格式
    messageDiv.innerHTML = marked.parse(text);

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


// 全局维护对话历史
const messages = [
      
];

function filterThoughts(text) {
    // 简单删除 <think>…</think> 区间
    return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
  }

// 发送消息处理
async function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    if (!message) return;

    // 1. 显示用户消息
    addMessage(message, true);
    input.value = '';

    // 2. 将这条用户消息加入历史
    messages.push({ role: 'user', content: message });

    // 3. 裁剪 messages，确保不会超出 token 限制
    const maxHistoryTokens = 8192; // 最大历史 token 数量
    while (JSON.stringify(messages).length > maxHistoryTokens) {
        // 移除最早的一条非 system 消息
        if (messages.length > 1) {
            messages.splice(1, 1);
        } else {
            break;
        }
    }

    // 4. 显示“正在输入”指示器
    const typingIndicator = showTypingIndicator();

    try {
        // 5. 调用 Chat Completion 接口
        const response = await fetch('http://127.0.0.1:11434/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                // 如果不需要认证，可移除 Authorization
                'Authorization': 'Bearer YOUR_API_KEY'
            },
            body: JSON.stringify({
                model: 'deepseek-r1:8b',
                messages: messages,
                temperature: 0.7,
                max_tokens: 16384,
            })
        });

        const data = await response.json();

        // 6. 移除“正在输入”指示器
        typingIndicator.remove();

        // 7. 从响应中提取机器人回复并展示（Markdown 格式）
        const botReply = filterThoughts(data.choices[0].message.content.trim());
        addMessage(botReply); // 这里会自动渲染 Markdown

        // 8. 将机器人回复加入历史
        messages.push({ role: 'assistant', content: botReply });

    } catch (error) {
        console.error('Error calling chat API:', error);
        typingIndicator.remove();
        addMessage('Sorry, there was an error sending your message. Please try again later.');
    }
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
