// Proxied through Next.js rewrite (avoids CORS for uploads/JSON requests)
const API_URL = '/api';

// Direct backend URL for SSE streaming (Next.js proxy buffers streams)
const BACKEND_URL = 'http://localhost:8000/api';

interface ChatHistory {
    role: string;
    content: string;
}

interface StreamCallbacks {
    onToken: (token: string) => void;
    onTool?: (tool: { tool: string; args: Record<string, unknown> }) => void;
    onCitations?: (citations: Array<{ source: string; locator: string; snippet: string }>) => void;
    onStatus?: (message: string) => void;
    onDone: () => void;
    onError?: (error: string) => void;
}

export const apiClient = {
    checkHealth: async () => {
        try {
            const res = await fetch(`${API_URL}/health`);
            return res.json();
        } catch (e) {
            return { status: 'offline' };
        }
    },

    getMemory: async () => {
        try {
            const res = await fetch(`${API_URL}/memory`);
            return res.json();
        } catch (e) {
            return { user: '', company: '' };
        }
    },

    getFileUrl: (filename: string) => {
        return `${API_URL}/files/${encodeURIComponent(filename)}`;
    },

    sendMessage: async (
        message: string,
        callbacks: StreamCallbacks,
        options?: { model?: string; history?: ChatHistory[] }
    ) => {
        try {
            // SSE streaming goes DIRECT to backend (not through Next.js proxy)
            const res = await fetch(`${BACKEND_URL}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message,
                    model: options?.model || 'openrouter',
                    history: options?.history || [],
                }),
            });

            if (!res.ok) {
                const errText = await res.text();
                callbacks.onError?.(`Server error: ${res.status} ${errText}`);
                callbacks.onDone();
                return;
            }

            if (!res.body) {
                callbacks.onError?.('No response body');
                callbacks.onDone();
                return;
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                // Normalize \r\n to \n (sse-starlette sends \r\n line endings)
                buffer = buffer.replace(/\r\n/g, '\n');
                const lines = buffer.split('\n\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    // SSE events may have multiple fields; extract data lines
                    const dataLines = line
                        .split('\n')
                        .filter(l => l.startsWith('data: '))
                        .map(l => l.slice(6));
                    const jsonStr = dataLines.join('\n');
                    if (!jsonStr) continue;

                    try {
                        const data = JSON.parse(jsonStr);
                        switch (data.type) {
                            case 'token':
                                callbacks.onToken(data.content);
                                break;
                            case 'tool':
                                callbacks.onTool?.({ tool: data.tool, args: data.args });
                                break;
                            case 'citations':
                                callbacks.onCitations?.(data.citations || []);
                                break;
                            case 'status':
                                callbacks.onStatus?.(data.content || '');
                                break;
                            case 'error':
                                callbacks.onError?.(data.content || 'Unknown error');
                                break;
                            case 'done':
                                callbacks.onDone();
                                return;
                        }
                    } catch (e) {
                        console.warn('SSE parse error:', e, 'chunk:', jsonStr);
                    }
                }
            }
            // Stream ended without explicit done event
            callbacks.onDone();
        } catch (e) {
            console.error('Chat error:', e);
            callbacks.onError?.(e instanceof Error ? e.message : 'Connection failed');
            callbacks.onDone();
        }
    },

    uploadFile: async (file: File): Promise<{ status: string; filename?: string; chunks?: number; message?: string }> => {
        const formData = new FormData();
        formData.append('file', file);
        let res: Response;
        try {
            res = await fetch(`${API_URL}/upload`, {
                method: 'POST',
                body: formData,
            });
        } catch (e) {
            throw new Error('Cannot reach backend. Is the server running on port 8000?');
        }
        if (!res.ok) {
            const errText = await res.text();
            throw new Error(`Upload failed (${res.status}): ${errText}`);
        }
        return res.json();
    },

    runWeather: async (query: string) => {
        try {
            const res = await fetch(`${API_URL}/tools/weather`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query }),
            });
            return res.json();
        } catch (e) {
            return { error: e instanceof Error ? e.message : 'Weather request failed' };
        }
    },

    runSandbox: async (code: string) => {
        try {
            const res = await fetch(`${API_URL}/tools/sandbox`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code }),
            });
            return res.json();
        } catch (e) {
            return { error: e instanceof Error ? e.message : 'Sandbox request failed' };
        }
    },
};
