"use client"

import * as React from "react"
import { useAppStore, type Citation, type ToolCall, type ThinkingStep } from "@/lib/store"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
    User, Bot, SendHorizonal, Loader2, Paperclip, FileText,
    ChevronDown, ChevronRight, Wrench, Clock, Search, Cloud,
    Code2, Brain, Zap, CheckCircle2
} from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { apiClient } from "@/lib/api"
import { toast } from "sonner"

const TOOL_ICONS: Record<string, React.ReactNode> = {
    search_documents: <Search className="w-3.5 h-3.5" />,
    get_weather: <Cloud className="w-3.5 h-3.5" />,
    execute_code: <Code2 className="w-3.5 h-3.5" />,
    write_memory: <Brain className="w-3.5 h-3.5" />,
}

const TOOL_LABELS: Record<string, string> = {
    search_documents: "Searching documents",
    get_weather: "Fetching weather",
    execute_code: "Running code",
    write_memory: "Writing memory",
}

function ThinkingBlock({ steps, isActive }: { steps: ThinkingStep[]; isActive: boolean }) {
    const [open, setOpen] = React.useState(true)

    // Auto-collapse when no longer active (answer arrived)
    React.useEffect(() => {
        if (!isActive && steps.length > 0) {
            const timer = setTimeout(() => setOpen(false), 600)
            return () => clearTimeout(timer)
        }
    }, [isActive, steps.length])

    if (!steps.length) return null

    return (
        <div className="mb-2">
            <button
                onClick={() => setOpen(!open)}
                className="flex items-center gap-1.5 px-2 py-1 text-xs font-mono text-orange-400/80 hover:text-orange-300 rounded transition-colors"
            >
                {isActive ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                    <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
                )}
                {open ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                <span>
                    {isActive ? "Thinking..." : `${steps.length} step${steps.length > 1 ? 's' : ''} completed`}
                </span>
            </button>

            <AnimatePresence>
                {open && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden"
                    >
                        <div className="ml-2 pl-3 border-l-2 border-orange-500/20 space-y-1 mt-1">
                            {steps.map((step, i) => (
                                <div key={i} className="flex items-center gap-2 text-xs font-mono text-zinc-500 py-0.5">
                                    {step.type === 'tool' ? (
                                        <Wrench className="w-3.5 h-3.5 text-orange-400/60 shrink-0" />
                                    ) : (
                                        <Zap className="w-3.5 h-3.5 text-zinc-600 shrink-0" />
                                    )}
                                    <span className="truncate">{step.content}</span>
                                </div>
                            ))}
                            {isActive && (
                                <div className="flex items-center gap-1 py-1">
                                    <span className="thinking-dot w-1 h-1 rounded-full bg-orange-400"></span>
                                    <span className="thinking-dot w-1 h-1 rounded-full bg-orange-400"></span>
                                    <span className="thinking-dot w-1 h-1 rounded-full bg-orange-400"></span>
                                </div>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}

function CitationsBlock({ citations, onCitationClick }: { citations: Citation[]; onCitationClick?: (c: Citation) => void }) {
    const [open, setOpen] = React.useState(false)
    if (!citations.length) return null
    return (
        <div className="mt-2 border border-orange-500/15 rounded-lg bg-orange-500/5">
            <button
                onClick={() => setOpen(!open)}
                className="flex items-center gap-1.5 px-3 py-2 text-xs font-mono text-orange-400/80 hover:text-orange-300 w-full transition-colors"
            >
                {open ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                <FileText className="w-3.5 h-3.5" />
                {citations.length} source{citations.length > 1 ? 's' : ''} cited
            </button>
            <AnimatePresence>
                {open && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden"
                    >
                        <div className="px-3 pb-2.5 space-y-2">
                            {citations.map((c, i) => (
                                <button
                                    key={i}
                                    onClick={() => onCitationClick?.(c)}
                                    className="w-full text-left citation-highlight text-xs font-mono rounded transition-all hover:bg-yellow-400/20 cursor-pointer"
                                >
                                    <span className="text-yellow-400 font-semibold">[{c.source}, {c.locator}]</span>
                                    <p className="text-zinc-500 mt-0.5 line-clamp-2">{c.snippet}</p>
                                </button>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}

function ToolCallsBlock({ tools }: { tools: ToolCall[] }) {
    if (!tools.length) return null
    return (
        <div className="mt-1 flex flex-wrap gap-1.5">
            {tools.map((tc, i) => (
                <span key={i} className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-mono bg-orange-500/10 text-orange-400 border border-orange-500/20 rounded-full">
                    {TOOL_ICONS[tc.tool] || <Wrench className="w-3 h-3" />}
                    {tc.tool}
                </span>
            ))}
        </div>
    )
}

function ResponseTimeBadge({ time }: { time: number }) {
    const formatted = time < 1000 ? `${time}ms` : `${(time / 1000).toFixed(1)}s`
    return (
        <span className="text-[10px] font-mono text-zinc-600 bg-zinc-800/50 px-1.5 py-0.5 rounded">
            <Clock className="w-3 h-3 inline mr-0.5 -mt-px" />
            {formatted}
        </span>
    )
}

export function ChatInterface() {
    const {
        messages, addMessage, updateLastMessage, isStreaming, setIsStreaming,
        addTokens, addTraceLog, addFile, model, setActiveSource, setActiveTab,
        setIsUploading, setUploadProgress, setMemory
    } = useAppStore()
    const [input, setInput] = React.useState("")
    const scrollRef = React.useRef<HTMLDivElement>(null)
    const fileInputRef = React.useRef<HTMLInputElement>(null)

    React.useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollIntoView({ behavior: "smooth" })
        }
    }, [messages])

    // Fetch memory on mount
    React.useEffect(() => {
        apiClient.getMemory().then(data => {
            setMemory(data.user || '', data.company || '')
        })
    }, [setMemory])

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files
        if (!files) return

        for (const file of Array.from(files)) {
            setIsUploading(true)
            setUploadProgress(`Uploading ${file.name}...`)
            toast.loading(`Uploading ${file.name}...`, { id: `upload-${file.name}` })
            addTraceLog(`Uploading ${file.name}...`)

            try {
                const result = await apiClient.uploadFile(file)
                if (result.status === 'success') {
                    addFile({ name: result.filename || file.name, chunks: result.chunks || 0, uploadedAt: Date.now() })
                    addTraceLog(`Indexed ${result.filename} (${result.chunks} chunks)`)

                    toast.success(`${result.filename} indexed`, {
                        id: `upload-${file.name}`,
                        description: `${result.chunks} chunks ready for search`,
                    })

                    addMessage({
                        role: 'system',
                        content: `${result.filename} uploaded and indexed (${result.chunks} chunks). You can now ask questions about it.`
                    })
                } else {
                    toast.info('File already indexed', { id: `upload-${file.name}` })
                    addTraceLog(`Upload skipped: ${result.message || 'already ingested'}`)
                }
            } catch (err) {
                toast.error(`Upload failed`, { id: `upload-${file.name}`, description: String(err) })
                addTraceLog(`Upload failed: ${err}`)
            } finally {
                setIsUploading(false)
                setUploadProgress('')
            }
        }
        if (fileInputRef.current) fileInputRef.current.value = ''
    }

    const handleCitationClick = (citation: Citation) => {
        setActiveSource(citation.source, messages.flatMap(m => m.citations || []))
        setActiveTab('source')
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!input.trim() || isStreaming) return

        const userMsg = input.trim()
        setInput("")

        const startTime = Date.now()
        addMessage({ role: "user", content: userMsg, startTime })
        addMessage({ role: "agent", content: "", isThinking: true, thinkingSteps: [], startTime })
        setIsStreaming(true)
        addTraceLog(`Query: "${userMsg}" [model=${model}]`)

        let fullResponse = ""
        let allCitations: Citation[] = []
        let allToolCalls: ToolCall[] = []
        let thinkingSteps: ThinkingStep[] = []

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
                    updateLastMessage({
                        content: fullResponse,
                        isThinking: false,
                        thinkingSteps: [...thinkingSteps],
                    })
                },
                onTool: (tool) => {
                    allToolCalls.push(tool)
                    const label = TOOL_LABELS[tool.tool] || tool.tool
                    const argSummary = tool.args.query
                        ? `"${String(tool.args.query).slice(0, 50)}"`
                        : tool.args.location
                        ? String(tool.args.location)
                        : ''
                    thinkingSteps.push({
                        type: 'tool',
                        content: `${label}${argSummary ? `: ${argSummary}` : ''}`,
                        timestamp: Date.now(),
                    })
                    addTraceLog(`Tool: ${tool.tool}(${JSON.stringify(tool.args).slice(0, 80)})`)
                    updateLastMessage({
                        toolCalls: [...allToolCalls],
                        thinkingSteps: [...thinkingSteps],
                    })
                },
                onStatus: (message) => {
                    thinkingSteps.push({
                        type: 'status',
                        content: message,
                        timestamp: Date.now(),
                    })
                    updateLastMessage({ thinkingSteps: [...thinkingSteps] })
                },
                onCitations: (citations) => {
                    allCitations = citations
                    addTraceLog(`Citations: ${citations.length} sources`)
                    updateLastMessage({ citations: [...allCitations] })
                },
                onDone: () => {
                    const elapsed = Date.now() - startTime
                    setIsStreaming(false)
                    updateLastMessage({
                        content: fullResponse,
                        isThinking: false,
                        citations: allCitations,
                        toolCalls: allToolCalls,
                        thinkingSteps,
                        responseTime: elapsed,
                    })
                    addTraceLog(`Response complete (${(elapsed / 1000).toFixed(1)}s, ${fullResponse.length} chars)`)

                    // Refresh memory after each response
                    apiClient.getMemory().then(data => {
                        setMemory(data.user || '', data.company || '')
                    })
                },
                onError: (err) => {
                    const elapsed = Date.now() - startTime
                    addTraceLog(`Error: ${err}`)
                    setIsStreaming(false)
                    updateLastMessage({
                        content: fullResponse || `Error: ${err}`,
                        isThinking: false,
                        responseTime: elapsed,
                    })
                },
            },
            { model, history }
        )
    }

    return (
        <div className="flex flex-col h-full bg-background relative">
            <ScrollArea className="flex-1 p-4">
                <div className="flex flex-col space-y-5 pb-24 max-w-3xl mx-auto">
                    {/* Welcome */}
                    {messages.length === 0 && (
                        <div className="text-center py-20 space-y-4">
                            <div className="relative inline-block">
                                <div className="w-16 h-16 rounded-2xl bg-orange-500/10 border border-orange-500/20 flex items-center justify-center mx-auto">
                                    <Zap className="w-8 h-8 text-orange-500" />
                                </div>
                            </div>
                            <h2 className="text-zinc-300 font-mono text-sm font-bold tracking-wide">AGENTIC RAG CHATBOT</h2>
                            <p className="text-zinc-600 font-mono text-xs max-w-md mx-auto leading-relaxed">
                                Upload documents and ask questions. I search, cite sources, fetch weather, run code, and remember key facts.
                            </p>
                            <div className="flex gap-2 justify-center mt-4">
                                {['RAG Search', 'Weather', 'Sandbox', 'Memory'].map(f => (
                                    <span key={f} className="text-[11px] font-mono px-2.5 py-1 rounded-full bg-orange-500/10 text-orange-400/70 border border-orange-500/15">
                                        {f}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}

                    <AnimatePresence initial={false}>
                        {messages.map((msg, i) => {
                            const isLastAgent = msg.role === 'agent' && i === messages.length - 1
                            const isThinkingNow = msg.isThinking && isStreaming && isLastAgent

                            if (msg.role === 'system') {
                                return (
                                    <motion.div
                                        key={i}
                                        initial={{ opacity: 0, y: 5 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="text-center"
                                    >
                                        <span className="text-xs font-mono text-zinc-500 bg-zinc-800/30 px-3 py-1.5 rounded-full border border-zinc-700/30">
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
                                    className="flex gap-3"
                                >
                                    <div className={`w-8 h-8 rounded-lg shrink-0 flex items-center justify-center border ${
                                        msg.role === 'user'
                                            ? 'border-zinc-700 bg-zinc-800/50 text-zinc-400'
                                            : 'border-orange-500/30 bg-orange-500/10 text-orange-500'
                                    }`}>
                                        {msg.role === 'user'
                                            ? <User className="w-4 h-4" />
                                            : <Bot className="w-4 h-4" />
                                        }
                                    </div>

                                    <div className="flex-1 space-y-1 mt-0.5 min-w-0">
                                        <div className="flex items-center gap-2">
                                            <span className={`text-xs font-bold font-mono uppercase ${
                                                msg.role === 'user' ? 'text-zinc-500' : 'text-orange-500'
                                            }`}>
                                                {msg.role === 'user' ? 'You' : 'Agent'}
                                            </span>
                                            {msg.responseTime && !isThinkingNow && (
                                                <ResponseTimeBadge time={msg.responseTime} />
                                            )}
                                        </div>

                                        {/* Thinking steps (collapsible) */}
                                        {msg.role === 'agent' && msg.thinkingSteps && msg.thinkingSteps.length > 0 && (
                                            <ThinkingBlock
                                                steps={msg.thinkingSteps}
                                                isActive={isThinkingNow || false}
                                            />
                                        )}

                                        {/* Tool call badges */}
                                        {msg.toolCalls && msg.toolCalls.length > 0 && !isThinkingNow && (
                                            <ToolCallsBlock tools={msg.toolCalls} />
                                        )}

                                        {/* Main content */}
                                        <div className="text-sm leading-relaxed text-zinc-200 font-mono whitespace-pre-wrap">
                                            {msg.content ? (
                                                <span>{msg.content}</span>
                                            ) : (
                                                isThinkingNow && !msg.thinkingSteps?.length && (
                                                    <span className="inline-flex items-center gap-1 text-zinc-600">
                                                        <Loader2 className="w-3.5 h-3.5 animate-spin text-orange-500" />
                                                        <span className="text-xs">Processing...</span>
                                                    </span>
                                                )
                                            )}
                                            {isThinkingNow && msg.content && (
                                                <span className="text-orange-500 animate-pulse">|</span>
                                            )}
                                        </div>

                                        {/* Citations */}
                                        {msg.citations && msg.citations.length > 0 && !isThinkingNow && (
                                            <CitationsBlock
                                                citations={msg.citations}
                                                onCitationClick={handleCitationClick}
                                            />
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
            <div className="p-4 bg-background/80 backdrop-blur-md border-t border-border">
                <form onSubmit={handleSubmit} className="max-w-3xl mx-auto relative">
                    {/* Hidden file input */}
                    <input
                        ref={fileInputRef}
                        type="file"
                        className="hidden"
                        accept=".pdf,.txt,.md,.html"
                        multiple
                        onChange={handleFileUpload}
                    />

                    <div className="glow-input rounded-xl overflow-hidden bg-zinc-900/50">
                        <div className="flex items-center">
                            <div className="pl-4 text-orange-500 pointer-events-none font-mono font-bold">
                                &gt;
                            </div>
                            <Input
                                autoFocus
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder={isStreaming ? "Agent is replying..." : "Ask about your documents, weather, or anything..."}
                                disabled={isStreaming}
                                className="border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 text-zinc-200 font-mono py-6 shadow-none placeholder:text-zinc-600"
                            />
                            <div className="pr-2 flex items-center gap-1">
                                <button
                                    type="button"
                                    onClick={() => fileInputRef.current?.click()}
                                    disabled={isStreaming}
                                    className="p-2 text-zinc-500 hover:text-orange-400 transition-colors disabled:opacity-30 rounded-lg hover:bg-orange-500/10"
                                    title="Upload documents (.pdf, .txt, .md, .html)"
                                >
                                    <Paperclip className="w-4 h-4" />
                                </button>
                                {isStreaming ? (
                                    <div className="p-2">
                                        <Loader2 className="w-4 h-4 text-orange-500 animate-spin" />
                                    </div>
                                ) : (
                                    <button
                                        type="submit"
                                        disabled={!input.trim()}
                                        className="p-2 text-zinc-500 hover:text-orange-400 disabled:opacity-30 transition-colors rounded-lg hover:bg-orange-500/10"
                                    >
                                        <SendHorizonal className="w-4 h-4" />
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className="text-center mt-2">
                        <span className="text-[10px] font-mono text-zinc-700">
                            Built by Sai Sudheer Naraharisetty for the CDF Hackathon
                        </span>
                    </div>
                </form>
            </div>
        </div>
    )
}
