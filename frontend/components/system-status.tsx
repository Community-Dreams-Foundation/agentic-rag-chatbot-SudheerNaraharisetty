"use client"

import * as React from "react"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { useAppStore } from "@/lib/store"
import { Database, UploadCloud, Loader2, Cpu, Brain, FileText, Zap } from "lucide-react"
import { apiClient } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { toast } from "sonner"

export function SystemStatus() {
    const {
        isConnected, setIsConnected, files, addFile, tokensUsed,
        model, setModel, messages, userMemory, companyMemory,
        setMemory, isUploading, setIsUploading, addTraceLog
    } = useAppStore()

    React.useEffect(() => {
        apiClient.checkHealth().then((data) => {
            if (data.status === "healthy") setIsConnected(true)
        })

        const interval = setInterval(() => {
            apiClient.checkHealth().then((data) => {
                setIsConnected(data.status === "healthy")
            })
        }, 10000)

        return () => clearInterval(interval)
    }, [setIsConnected])

    // Fetch memory periodically
    React.useEffect(() => {
        const fetchMemory = () => {
            apiClient.getMemory().then(data => {
                setMemory(data.user || '', data.company || '')
            })
        }
        fetchMemory()
        const interval = setInterval(fetchMemory, 15000)
        return () => clearInterval(interval)
    }, [setMemory])

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0]
            setIsUploading(true)
            toast.loading(`Uploading ${file.name}...`, { id: `sidebar-upload-${file.name}` })
            try {
                const res = await apiClient.uploadFile(file)
                if (res.status === 'success') {
                    addFile({ name: res.filename || file.name, chunks: res.chunks || 0, uploadedAt: Date.now() })
                    toast.success(`${res.filename} indexed`, {
                        id: `sidebar-upload-${file.name}`,
                        description: `${res.chunks} chunks ready`,
                    })
                    addTraceLog(`Indexed ${res.filename} (${res.chunks} chunks)`)
                } else {
                    toast.info('Already indexed', { id: `sidebar-upload-${file.name}` })
                }
            } catch (error) {
                toast.error('Upload failed', { id: `sidebar-upload-${file.name}` })
            } finally {
                setIsUploading(false)
            }
        }
    }

    const memoryLines = (userMemory || '').split('\n').filter(l => l.trim()).length +
                       (companyMemory || '').split('\n').filter(l => l.trim()).length

    return (
        <div className="flex flex-col h-full bg-background border-r border-border">
            {/* Header */}
            <div className="px-5 py-4 border-b border-border">
                <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg bg-orange-500/10 border border-orange-500/20 flex items-center justify-center">
                        <Zap className="w-5 h-5 text-orange-500" />
                    </div>
                    <div>
                        <h2 className="text-sm font-bold tracking-wide text-zinc-200 font-mono">AGENTIC RAG</h2>
                        <div className="flex items-center gap-1.5 mt-0.5">
                            <div className={`w-2 h-2 rounded-full ${
                                isConnected
                                    ? 'bg-green-500 shadow-[0_0_6px_rgba(34,197,94,0.5)]'
                                    : 'bg-red-500 shadow-[0_0_6px_rgba(239,68,68,0.5)]'
                            }`} />
                            <span className="text-[11px] text-zinc-500 font-mono">
                                {isConnected ? 'CONNECTED' : 'OFFLINE'}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Model Selection */}
            <div className="px-5 py-4 border-b border-border">
                <h3 className="text-[11px] uppercase tracking-widest text-zinc-500 mb-3 flex items-center gap-1.5 font-mono">
                    <Cpu className="w-3.5 h-3.5" />
                    Model
                </h3>
                <div className="flex gap-2">
                    <button
                        onClick={() => setModel('openrouter')}
                        className={`flex-1 text-xs font-mono py-2 px-3 rounded-lg border transition-all ${
                            model === 'openrouter'
                                ? 'border-orange-500/40 bg-orange-500/10 text-orange-400 shadow-[0_0_10px_rgba(249,115,22,0.1)]'
                                : 'border-zinc-800 text-zinc-500 hover:border-zinc-700 hover:text-zinc-400'
                        }`}
                    >
                        Llama 3.3 70B
                    </button>
                    <button
                        onClick={() => setModel('groq')}
                        className={`flex-1 text-xs font-mono py-2 px-3 rounded-lg border transition-all ${
                            model === 'groq'
                                ? 'border-orange-500/40 bg-orange-500/10 text-orange-400 shadow-[0_0_10px_rgba(249,115,22,0.1)]'
                                : 'border-zinc-800 text-zinc-500 hover:border-zinc-700 hover:text-zinc-400'
                        }`}
                    >
                        Groq Fast
                    </button>
                </div>
            </div>

            {/* Upload & Knowledge Base */}
            <div className="flex-1 overflow-hidden flex flex-col min-h-0">
                <div className="px-5 py-4">
                    <Button
                        variant="outline"
                        className="w-full border-dashed border-orange-500/20 hover:border-orange-500/40 hover:bg-orange-500/5 text-zinc-400 text-xs font-mono h-10"
                        onClick={() => document.getElementById('sidebar-file-upload')?.click()}
                        disabled={isUploading}
                    >
                        {isUploading ? <Loader2 className="w-4 h-4 mr-2 animate-spin text-orange-400" /> : <UploadCloud className="w-4 h-4 mr-2" />}
                        {isUploading ? "Indexing..." : "Upload Document"}
                    </Button>
                    <input
                        id="sidebar-file-upload"
                        type="file"
                        className="hidden"
                        onChange={handleUpload}
                        accept=".pdf,.txt,.md,.html"
                    />
                </div>

                <Separator className="bg-border" />

                <div className="px-5 py-4 flex-1 overflow-hidden flex flex-col min-h-0">
                    <h3 className="text-[11px] uppercase tracking-widest text-zinc-500 mb-3 flex items-center gap-1.5 font-mono shrink-0">
                        <Database className="w-3.5 h-3.5" />
                        Knowledge Base
                    </h3>
                    <ScrollArea className="flex-1 min-h-0">
                        {files.length === 0 ? (
                            <div className="py-8 text-center text-zinc-600 text-xs italic font-mono">
                                No documents indexed
                            </div>
                        ) : (
                            <div className="space-y-1.5">
                                {files.map((f, i) => (
                                    <div key={i} className="flex items-center justify-between p-2.5 rounded-lg bg-zinc-900/40 border border-zinc-800/50">
                                        <div className="flex items-center gap-2 min-w-0">
                                            <FileText className="w-3.5 h-3.5 text-orange-400/60 shrink-0" />
                                            <span className="text-xs text-zinc-300 truncate font-mono">{f.name}</span>
                                        </div>
                                        <Badge variant="secondary" className="text-[10px] h-5 bg-orange-500/10 text-orange-400/80 border-orange-500/20 ml-2 shrink-0">
                                            {f.chunks} chunks
                                        </Badge>
                                    </div>
                                ))}
                            </div>
                        )}
                    </ScrollArea>
                </div>

                <Separator className="bg-border" />

                {/* Memory Summary */}
                <div className="px-5 py-4 shrink-0">
                    <h3 className="text-[11px] uppercase tracking-widest text-zinc-500 mb-3 flex items-center gap-1.5 font-mono">
                        <Brain className="w-3.5 h-3.5 text-orange-500/60" />
                        Persistent Memory
                    </h3>
                    <div className="space-y-2">
                        <div className="p-3 bg-zinc-900/40 rounded-lg border border-zinc-800/50">
                            <div className="text-[11px] text-zinc-500 font-mono mb-1">USER_MEMORY.md</div>
                            <div className="text-xs font-mono text-zinc-400 line-clamp-2 leading-relaxed">
                                {userMemory ? userMemory.split('\n').filter(l => l.trim()).slice(0, 2).join(' | ') : 'Empty'}
                            </div>
                        </div>
                        <div className="p-3 bg-zinc-900/40 rounded-lg border border-zinc-800/50">
                            <div className="text-[11px] text-zinc-500 font-mono mb-1">COMPANY_MEMORY.md</div>
                            <div className="text-xs font-mono text-zinc-400 line-clamp-2 leading-relaxed">
                                {companyMemory ? companyMemory.split('\n').filter(l => l.trim()).slice(0, 2).join(' | ') : 'Empty'}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Metrics Footer */}
            <div className="px-5 py-4 border-t border-border shrink-0">
                <div className="grid grid-cols-3 gap-2">
                    <div className="p-2.5 bg-zinc-900/40 rounded-lg border border-zinc-800/50 text-center">
                        <div className="text-[10px] text-zinc-500 font-mono uppercase">Tokens</div>
                        <div className="text-sm font-mono text-zinc-300 mt-0.5 tabular-nums">{tokensUsed.toLocaleString()}</div>
                    </div>
                    <div className="p-2.5 bg-zinc-900/40 rounded-lg border border-zinc-800/50 text-center">
                        <div className="text-[10px] text-zinc-500 font-mono uppercase">Msgs</div>
                        <div className="text-sm font-mono text-zinc-300 mt-0.5 tabular-nums">{messages.length}</div>
                    </div>
                    <div className="p-2.5 bg-zinc-900/40 rounded-lg border border-zinc-800/50 text-center">
                        <div className="text-[10px] text-zinc-500 font-mono uppercase">Memory</div>
                        <div className="text-sm font-mono text-zinc-300 mt-0.5 tabular-nums">{memoryLines}</div>
                    </div>
                </div>
            </div>
        </div>
    )
}
