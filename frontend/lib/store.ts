import { create } from 'zustand';

export interface Citation {
    source: string;
    locator: string;
    snippet: string;
}

export interface ToolCall {
    tool: string;
    args: Record<string, unknown>;
}

export interface ChatMessage {
    role: 'user' | 'agent' | 'system';
    content: string;
    isThinking?: boolean;
    citations?: Citation[];
    toolCalls?: ToolCall[];
}

interface AppState {
    // Chat State
    messages: ChatMessage[];
    addMessage: (msg: ChatMessage) => void;
    updateLastMessage: (updates: Partial<ChatMessage>) => void;
    isStreaming: boolean;
    setIsStreaming: (is: boolean) => void;

    // System Context
    isConnected: boolean;
    setIsConnected: (status: boolean) => void;
    files: string[];
    addFile: (filename: string) => void;
    tokensUsed: number;
    addTokens: (count: number) => void;

    // Inspector State
    activeTab: 'trace' | 'source' | 'visual';
    setActiveTab: (tab: 'trace' | 'source' | 'visual') => void;

    // Real-time Logs
    traceLogs: string[];
    addTraceLog: (log: string) => void;

    // Model selection
    model: 'openrouter' | 'groq';
    setModel: (model: 'openrouter' | 'groq') => void;
}

export const useAppStore = create<AppState>((set) => ({
    messages: [],
    addMessage: (msg) => set((state) => ({ messages: [...state.messages, msg] })),
    updateLastMessage: (updates) => set((state) => {
        const newMessages = [...state.messages];
        if (newMessages.length > 0) {
            const last = newMessages[newMessages.length - 1];
            newMessages[newMessages.length - 1] = { ...last, ...updates };
        }
        return { messages: newMessages };
    }),
    isStreaming: false,
    setIsStreaming: (is) => set({ isStreaming: is }),

    isConnected: false,
    setIsConnected: (status) => set({ isConnected: status }),
    files: [],
    addFile: (filename) => set((state) => ({
        files: state.files.includes(filename) ? state.files : [...state.files, filename]
    })),
    tokensUsed: 0,
    addTokens: (count) => set((state) => ({ tokensUsed: state.tokensUsed + count })),

    activeTab: 'trace',
    setActiveTab: (tab) => set({ activeTab: tab }),

    traceLogs: [],
    addTraceLog: (log) => set((state) => ({
        traceLogs: [...state.traceLogs, `[${new Date().toLocaleTimeString()}] ${log}`]
    })),

    model: 'openrouter',
    setModel: (model) => set({ model }),
}));
