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

export interface ThinkingStep {
    type: 'tool' | 'status';
    content: string;
    timestamp: number;
}

export interface ChatMessage {
    role: 'user' | 'agent' | 'system';
    content: string;
    isThinking?: boolean;
    citations?: Citation[];
    toolCalls?: ToolCall[];
    thinkingSteps?: ThinkingStep[];
    responseTime?: number; // milliseconds
    startTime?: number;
}

export interface UploadedFile {
    name: string;
    chunks: number;
    uploadedAt: number;
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
    files: UploadedFile[];
    addFile: (file: UploadedFile) => void;
    tokensUsed: number;
    addTokens: (count: number) => void;

    // Inspector State
    activeTab: 'trace' | 'source' | 'memory' | 'tools';
    setActiveTab: (tab: 'trace' | 'source' | 'memory' | 'tools') => void;

    // Real-time Logs
    traceLogs: string[];
    addTraceLog: (log: string) => void;

    // Model selection
    model: 'openrouter' | 'groq';
    setModel: (model: 'openrouter' | 'groq') => void;

    // Memory content (for judges)
    userMemory: string;
    companyMemory: string;
    setMemory: (user: string, company: string) => void;

    // Active source for PDF viewer
    activeSource: string | null;
    activeCitations: Citation[];
    setActiveSource: (source: string | null, citations?: Citation[]) => void;

    // Upload state
    isUploading: boolean;
    setIsUploading: (is: boolean) => void;
    uploadProgress: string;
    setUploadProgress: (msg: string) => void;
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
    addFile: (file) => set((state) => ({
        files: state.files.some(f => f.name === file.name) ? state.files : [...state.files, file]
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

    userMemory: '',
    companyMemory: '',
    setMemory: (user, company) => set({ userMemory: user, companyMemory: company }),

    activeSource: null,
    activeCitations: [],
    setActiveSource: (source, citations = []) => set({ activeSource: source, activeCitations: citations }),

    isUploading: false,
    setIsUploading: (is) => set({ isUploading: is }),
    uploadProgress: '',
    setUploadProgress: (msg) => set({ uploadProgress: msg }),
}));
