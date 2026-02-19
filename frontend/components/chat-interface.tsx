"use client"

import * as React from "react"
import { useAppStore, type Citation, type ToolCall } from "@/lib/store"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { User, Bot, SendHorizonal, Loader2, Paperclip, FileText, ChevronDown, ChevronRight, Wrench } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { apiClient } from "@/lib/api"

function CitationsBlock({ citations }: { citations: Citation[] }) {
    const [open, setOpen] = React.useState(false)
    if (!citations.length) return null
    return (
        <div className="mt-2 border border-zinc-800 rounded">
            <button
                onClick={() => setOpen(!open)}
                className="flex items-center gap-1 px-2 py-1 text-[11px] font-mono text-amber-500/80 hover:text-amber-400 w-full"
            >
                {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                {citations.length} source{citations.length > 1 ? 's' : ''}
            </button>
            {open && (
                <div className="px-2 pb-2 space-y-1">
                    {citations.map((c, i) => (
                        <div key={i} className="text-[11px] font-mono text-zinc-500 border-l-2 border-amber-500/30 pl-2">
                            <span className="text-zinc-400">[{c.source}, {c.locator}]</span>
                            <p className="text-zinc-600 mt-0.5">{c.snippet.slice(0, 150)}...</p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}

function ToolCallsBlock({ tools }: { tools: ToolCall[] }) {
    if (!tools.length) return null
    return (
        <div className="mt-1 flex flex-wrap gap-1">
            {tools.map((tc, i) => (
                <span key={i} className="inline-flex items-center gap-1 px-1.5 py-0.5 text-[10px] font-mono bg-blue-500/10 text-blue-400 border border-blue-500/20 rounded">
                    <Wrench className="w-2.5 h-2.5" />
                    {tc.tool}
                </span>
            ))}
        </div>
    )
}

export function ChatInterface() {
    const { messages, addMessage, updateLastMessage, isStreaming, setIsStreaming, addTokens, addTraceLog, addFile, model } = useAppStore()
    const [input, setInput] = React.useState("")
    const scrollRef = React.useRef<HTMLDivElement>(null)
    const fileInputRef = React.useRef<HTMLInputElement>(null)

    React.useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollIntoView({ behavior: "smooth" })
        }
    }, [messages])

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files
        if (!files) return

        for (const file of Array.from(files)) {
            addTraceLog(`Uploading ${file.name}...`)
            try {
                const result = await apiClient.uploadFile(file)
                if (result.status === 'success') {
                    addFile(result.filename || file.name)
                    addTraceLog(`Indexed ${result.filename} (${result.chunks} chunks)`)
                    addMessage({
                        role: 'system',
                        content: `📄 **${result.filename}** uploaded and indexed (${result.chunks} chunks). You can now ask questions about it.`
                    })
                } else {
                    addTraceLog(`Upload skipped: ${result.message || 'already ingested'}`)
                }
            } catch (err) {
                addTraceLog(`Upload failed: ${err}`)
            }
        }
        // Reset input
        if (fileInputRef.current) fileInputRef.current.value = ''
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!input.trim() || isStreaming) return

        const userMsg = input.trim()
        setInput("")

        addMessage({ role: "user", content: userMsg })
        addMessage({ role: "agent", content: "", isThinking: true })
        setIsStreaming(true)
        addTraceLog(`Query: "${userMsg}" [model=${model}]`)

        let fullResponse = ""
        let allCitations: Citation[] = []
        let allToolCalls: ToolCall[] = []

        // Build chat history (exclude current + placeholder)
        const history = messages
            .filter(m => m.role !== 'system')
            .slice(-6)
            .map(m => ({
                role: m.role === 'agent' ? 'assistant' : 'user',
                content: m.content
            }))

        await apiClient.sendMessage(
            userMsg,
            {
                onToken: (token) => {
                    fullResponse += token
                    addTokens(1)
                    updateLastMessage({ content: fullResponse, isThinking: false })
                },
                onTool: (tool) => {
                    allToolCalls.push(tool)
                    addTraceLog(`Tool: ${tool.tool}(${JSON.stringify(tool.args).slice(0, 80)})`)
                    updateLastMessage({ toolCalls: [...allToolCalls] })
                },
                onCitations: (citations) => {
                    allCitations = citations
                    addTraceLog(`Citations: ${citations.length} sources`)
                    updateLastMessage({ citations: [...allCitations] })
                },
                onDone: () => {
                    setIsStreaming(false)
                    updateLastMessage({
                        content: fullResponse,
                        isThinking: false,
                        citations: allCitations,
                        toolCalls: allToolCalls,
                    })
                    addTraceLog(`Response complete (${fullResponse.length} chars)`)
                },
                onError: (err) => {
                    addTraceLog(`Error: ${err}`)
                    updateLastMessage({ content: fullResponse || `Error: ${err}`, isThinking: false })
                },
            },
            { model, history }
        )
    }

    return (
        <div className="flex flex-col h-full bg-black relative">
            <ScrollArea className="flex-1 p-4">
                <div className="flex flex-col space-y-6 pb-20 max-w-3xl mx-auto">
                    {/* Welcome message */}
                    {messages.length === 0 && (
                        <div className="text-center py-16 space-y-3">
                            <div className="text-amber-500/50 text-4xl">⚡</div>
                            <h2 className="text-zinc-400 font-mono text-sm">AGENTIC RAG CHATBOT</h2>
                            <p className="text-zinc-600 font-mono text-xs max-w-md mx-auto">
                                Upload documents and ask questions. I'll search, cite sources, fetch weather, run code, and remember key facts.
                            </p>
                        </div>
                    )}

                    <AnimatePresence initial={false}>
                        {messages.map((msg, i) => {
                            const isThinkingNow = msg.isThinking && isStreaming && i === messages.length - 1

                            if (msg.role === 'system') {
                                return (
                                    <motion.div
                                        key={i}
                                        initial={{ opacity: 0, y: 5 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="text-center"
                                    >
                                        <span className="text-[11px] font-mono text-zinc-600 bg-zinc-900/50 px-3 py-1 rounded-full">
                                            {msg.content}
                                        </span>
                                    </motion.div>
                                )
                            }

                            return (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="flex gap-4"
                                >
                                    <div className={`w-8 h-8 rounded shrink-0 flex items-center justify-center border ${
                                        msg.role === 'user'
                                            ? 'border-zinc-700 bg-zinc-900 text-zinc-400'
                                            : 'border-amber-500/20 bg-amber-500/10 text-amber-500'
                                    }`}>
                                        {msg.role === 'user'
                                            ? <User className="w-4 h-4" />
                                            : <Bot className="w-4 h-4" />
                                        }
                                    </div>

                                    <div className="flex-1 space-y-1 mt-1 max-w-[90%]">
                                        <div className="flex items-center gap-2">
                                            <span className={`text-xs font-bold font-mono uppercase ${
                                                msg.role === 'user' ? 'text-zinc-500' : 'text-amber-500'
                                            }`}>
                                                {msg.role === 'user' ? 'You' : 'Agent'}
                                            </span>
                                            {isThinkingNow && !msg.content && (
                                                <span className="text-[10px] text-zinc-600 animate-pulse font-mono">
                                                    {msg.toolCalls?.length ? `Using ${msg.toolCalls[msg.toolCalls.length-1].tool}...` : 'Thinking...'}
                                                </span>
                                            )}
                                        </div>

                                        {/* Tool calls */}
                                        {msg.toolCalls && <ToolCallsBlock tools={msg.toolCalls} />}

                                        {/* Main content */}
                                        <div className="text-sm leading-relaxed text-zinc-100 font-mono whitespace-pre-wrap">
                                            {msg.content ? (
                                                <span className="text-zinc-300">{msg.content}</span>
                                            ) : (
                                                isThinkingNow && (
                                                    <span className="inline-flex items-center gap-1 text-zinc-700">
                                                        <Loader2 className="w-3 h-3 animate-spin" />
                                                    </span>
                                                )
                                            )}
                                            {isThinkingNow && msg.content && (
                                                <span className="text-amber-500 animate-pulse">▌</span>
                                            )}
                                        </div>

                                        {/* Citations */}
                                        {msg.citations && !isThinkingNow && (
                                            <CitationsBlock citations={msg.citations} />
                                        )}
                                    </div>
                                </motion.div>
                            )
                        })}
                    </AnimatePresence>
                    <div ref={scrollRef} />
                </div>
            </ScrollArea>

            {/* Input Area */}
            <div className="p-4 bg-black/80 backdrop-blur border-t border-zinc-800">
                <form onSubmit={handleSubmit} className="max-w-3xl mx-auto relative group">
                    <div className="absolute left-4 top-3 text-amber-600 pointer-events-none font-mono">❯</div>

                    {/* Hidden file input */}
                    <input
                        ref={fileInputRef}
                        type="file"
                        className="hidden"
                        accept=".pdf,.txt,.md,.html"
                        multiple
                        onChange={handleFileUpload}
                    />

                    <Input
                        autoFocus
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder={isStreaming ? "Agent is replying..." : "Ask about your documents, weather, or anything..."}
                        disabled={isStreaming}
                        className="pl-8 pr-20 bg-zinc-950/50 border-zinc-800 focus-visible:ring-1 focus-visible:ring-amber-500/50 text-zinc-200 font-mono py-6 shadow-2xl"
                    />
                    <div className="absolute right-2 top-2 flex items-center gap-1">
                        <button
                            type="button"
                            onClick={() => fileInputRef.current?.click()}
                            disabled={isStreaming}
                            className="p-2 text-zinc-500 hover:text-amber-500 transition-colors disabled:opacity-30"
                            title="Upload documents"
                        >
                            <Paperclip className="w-4 h-4" />
                        </button>
                        {isStreaming ? (
                            <Loader2 className="w-5 h-5 text-zinc-600 animate-spin mt-0 mr-1" />
                        ) : (
                            <button type="submit" className="p-2 text-zinc-500 hover:text-amber-500 transition-colors">
                                <SendHorizonal className="w-4 h-4" />
                            </button>
                        )}
                    </div>
                </form>
            </div>
        </div>
    )
}
