/**
 * main.js - 聊天交互逻辑
 */

// 全局变量
let messageCount = 0;
const REMINDER_INTERVAL = 10; // 每10条消息提醒一次
const messages = [
    { 
        role: "system", 
        content: "Reminder: Wrap all actions with double bracket (like this) ."
    },
    {
        role: 'system',
        content: 'Reminder: Wrap all narration or actions outside dialogue with double asterisks **like this**.'
    },
    {
        role: 'system',
        content: 'Reminder: Wrap all internal thoughts, feelings outside dialogue with double backticks `like this`.'
    },
    { 
        role: "system", 
        content: "Reminder: Never break character or deviate from the format. Never reveal you are an AI or break the fourth wall."
    },

];

// 添加消息到聊天框（支持 Markdown 格式）
function addMessage(text, isUser = false) {
    const chatBox = document.getElementById('chat-box');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.innerHTML = marked.parse(text);

    // 添加淡入动画
    messageDiv.style.opacity = '0';
    chatBox.appendChild(messageDiv);
    setTimeout(() => {
        messageDiv.style.opacity = '1';
    }, 10);

    // 自动滚动到底部
    chatBox.scrollTo({
        top: chatBox.scrollHeight,
        behavior: 'smooth'
    });
}

// 显示“正在输入”指示器
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

// 加载角色配置
async function loadCharacterConfig(role) {
    try {
        const response = await fetch('./js/characters.json');
        if (!response.ok) {
            throw new Error('Failed to load character config');
        }
        const characters = await response.json();
        const character = characters[role];
        if (!character) {
            throw new Error(`Character ${role} not found`);
        }
        return character;
    } catch (error) {
        console.error('Error loading character config:', error);
        return {
            name: 'Default',
            age: 0,
            gender: 'Unknown',
            personality: 'Generic character'
        };
    }
}

// 加载系统提示
async function loadSystemPrompt(filePath) {
    try {
        const response = await fetch(filePath);
        if (!response.ok) {
            throw new Error(`Failed to load system prompt from ${filePath}`);
        }
        return await response.text();
    } catch (error) {
        console.error('Error loading system prompt:', error);
        throw error;
    }
}

// 简单删除 <think>…</think> 区间
function filterThoughts(text) {
    return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
  }

// 发送消息
async function sendMessage() {
    const input = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn'); // 获取发送按钮
    const message = input.value.trim();
    if (!message) return;

    // 禁用发送按钮并应用禁用样式
    sendBtn.disabled = true;

    // 显示用户消息
    addMessage(message, true);
    input.value = '';

    // 将用户消息加入历史
    messages.push({ role: 'user', content: message });
    console.log('用户发送消息:', message);

    // 裁剪消息历史，防止超出 token 限制
    const maxHistoryTokens = 8192;
    while (JSON.stringify(messages).length > maxHistoryTokens) {
        if (messages.length > 1) {
            messages.splice(1, 1);
        } else {
            break;
        }
    }

    // 显示“正在输入”指示器
    const typingIndicator = showTypingIndicator();

    try {
        // 调用 Chat Completion 接口
        const response = await fetch('http://127.0.0.1:11434/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'deepseek-r1:8b',
                messages: messages,
                temperature: 0.7,
                max_tokens: 16384,
            })
        });

        const data = await response.json();

        // 移除“正在输入”指示器
        typingIndicator.remove();

        // 显示机器人回复
        const botReply = filterThoughts(data.choices[0].message.content.trim());
        addMessage(botReply);
        messages.push({ role: 'assistant', content: botReply });
        console.log('机器人回复:', botReply);   

        // 每 N 条消息发送一次提醒
        messageCount++;
        if (messageCount % REMINDER_INTERVAL === 0) {
            messages.push({ 
                role: 'system', 
                content: "Reminder: Maintain role settings, avoid breaking character and pay attention to formatting requirements"
            });
        }
    } catch (error) {
        console.error('Error calling chat API:', error);
        typingIndicator.remove();
        addMessage('Sorry, there was an error sending your message. Please try again later.');
    } finally {
        // 启用发送按钮并恢复样式
        sendBtn.disabled = false;
    }
}

// 初始化事件绑定
function initializeEventListeners() {
    const input = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const exportBtn = document.getElementById('export-btn');

    // 回车发送消息
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // 输入框高度自适应
    input.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight - 20 + 'px';
    });

    // 发送按钮点击事件
    sendBtn.addEventListener('click', sendMessage);

    // 导出聊天记录
    exportBtn.addEventListener('click', () => {
        const blob = new Blob([JSON.stringify(messages, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'chat_log.json';
        a.click();
        URL.revokeObjectURL(url);
    });
}

// 初始化角色选择逻辑
function initializeRoleSelection() {
    const roleModal = document.getElementById('role-modal');
    const modalRoleSelect = document.getElementById('modal-role-select');
    const confirmRoleBtn = document.getElementById('confirm-role-btn');
    const messageInput = document.getElementById('message-input');

    // 显示模态窗口
    roleModal.style.display = 'block';

    // 确认角色选择
    confirmRoleBtn.addEventListener('click', async () => {
        console.log('用户选择了角色：', modalRoleSelect.value);
        const selectedRole = modalRoleSelect.value;

        try {
            // 加载角色配置和系统提示
            const promptPath = `./prompts/${selectedRole}.txt`;
            const [loadedPrompt, loadedCharacter] = await Promise.all([
                loadSystemPrompt(promptPath),
                loadCharacterConfig(selectedRole)
            ]);

            systemPrompt = loadedPrompt;
            character = loadedCharacter;

            // 更新输入框占位符
            messageInput.placeholder = `与${character.name}互动吧~`;

            // 初始化聊天历史
            const characterInfo = `characterInfo:
name: ${character.name}
age: ${character.age}
gender: ${character.gender}
personality: ${character.personality}`;
            messages.unshift({ role: 'system', content: `${characterInfo}\n\n${systemPrompt}` });

            // 隐藏模态窗口
            roleModal.style.display = 'none';

            // 角色输出第一句问候语
            const greeting = character.greeting || `你好，我是${character.name}，很高兴与你交流！`;
            addMessage(greeting);
            messages.push({ role: 'assistant', content: greeting });
        } catch (error) {
            console.error('加载角色配置或系统提示失败:', error);
        }
    });
}

// 初始化程序
document.addEventListener('DOMContentLoaded', () => {
    console.log('main.js 已加载');
    initializeEventListeners();
    initializeRoleSelection();
});
