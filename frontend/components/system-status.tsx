"use client"

import * as React from "react"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { useAppStore } from "@/lib/store"
import { Activity, Database, Server, UploadCloud, Loader2, Cpu, Brain, FileText, Zap } from "lucide-react"
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
            <div className="p-4 border-b border-border">
                <div className="flex items-center gap-2">
                    <div className="w-7 h-7 rounded-lg bg-orange-500/10 border border-orange-500/20 flex items-center justify-center">
                        <Zap className="w-4 h-4 text-orange-500" />
                    </div>
                    <div>
                        <h2 className="text-[11px] font-bold tracking-wider text-zinc-300 font-mono">AGENTIC RAG</h2>
                        <div className="flex items-center gap-1.5 mt-0.5">
                            <div className={`w-1.5 h-1.5 rounded-full ${
                                isConnected
                                    ? 'bg-green-500 shadow-[0_0_6px_rgba(34,197,94,0.5)]'
                                    : 'bg-red-500 shadow-[0_0_6px_rgba(239,68,68,0.5)]'
                            }`} />
                            <span className="text-[9px] text-zinc-600 font-mono">
                                {isConnected ? 'CONNECTED' : 'OFFLINE'}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Model Selection */}
            <div className="p-3 border-b border-border">
                <h3 className="text-[9px] uppercase tracking-wider text-zinc-600 mb-2 flex items-center gap-1 font-mono">
                    <Cpu className="w-3 h-3" />
                    Model
                </h3>
                <div className="flex gap-1">
                    <button
                        onClick={() => setModel('openrouter')}
                        className={`flex-1 text-[10px] font-mono py-1.5 px-2 rounded-md border transition-all ${
                            model === 'openrouter'
                                ? 'border-orange-500/40 bg-orange-500/10 text-orange-400 shadow-[0_0_8px_rgba(249,115,22,0.1)]'
                                : 'border-zinc-800 text-zinc-600 hover:border-zinc-700'
                        }`}
                    >
                        Llama 3.3 70B
                    </button>
                    <button
                        onClick={() => setModel('groq')}
                        className={`flex-1 text-[10px] font-mono py-1.5 px-2 rounded-md border transition-all ${
                            model === 'groq'
                                ? 'border-orange-500/40 bg-orange-500/10 text-orange-400 shadow-[0_0_8px_rgba(249,115,22,0.1)]'
                                : 'border-zinc-800 text-zinc-600 hover:border-zinc-700'
                        }`}
                    >
                        Groq Fast
                    </button>
                </div>
            </div>

            {/* Upload & Files */}
            <div className="flex-1 overflow-hidden flex flex-col">
                <div className="p-3">
                    <Button
                        variant="outline"
                        className="w-full border-dashed border-orange-500/20 hover:border-orange-500/40 hover:bg-orange-500/5 text-zinc-400 text-[11px] font-mono"
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

                <div className="p-2 flex-1 overflow-hidden">
                    <h3 className="text-[9px] uppercase tracking-wider text-zinc-600 mb-2 px-2 flex items-center gap-1 font-mono">
                        <Database className="w-3 h-3" />
                        Knowledge Base
                    </h3>
                    <ScrollArea className="h-[140px]">
                        {files.length === 0 ? (
                            <div className="px-4 py-6 text-center text-zinc-700 text-[10px] italic font-mono">
                                No documents indexed
                            </div>
                        ) : (
                            <div className="space-y-1 px-2">
                                {files.map((f, i) => (
                                    <div key={i} className="flex items-center justify-between p-2 rounded-md bg-zinc-900/30 border border-zinc-800/30">
                                        <div className="flex items-center gap-1.5 min-w-0">
                                            <FileText className="w-3 h-3 text-orange-400/50 shrink-0" />
                                            <span className="text-[10px] text-zinc-300 truncate font-mono">{f.name}</span>
                                        </div>
                                        <Badge variant="secondary" className="text-[8px] h-4 bg-orange-500/10 text-orange-400/70 border-orange-500/15">
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
                <div className="p-3">
                    <h3 className="text-[9px] uppercase tracking-wider text-zinc-600 mb-2 flex items-center gap-1 font-mono">
                        <Brain className="w-3 h-3 text-orange-500/60" />
                        Persistent Memory
                    </h3>
                    <div className="space-y-1.5">
                        <div className="p-2 bg-zinc-900/30 rounded-md border border-zinc-800/30">
                            <div className="text-[9px] text-zinc-600 font-mono">USER_MEMORY.md</div>
                            <div className="text-[10px] font-mono text-zinc-400 mt-0.5 line-clamp-2">
                                {userMemory ? userMemory.split('\n').filter(l => l.trim()).slice(0, 2).join(' | ') : 'Empty'}
                            </div>
                        </div>
                        <div className="p-2 bg-zinc-900/30 rounded-md border border-zinc-800/30">
                            <div className="text-[9px] text-zinc-600 font-mono">COMPANY_MEMORY.md</div>
                            <div className="text-[10px] font-mono text-zinc-400 mt-0.5 line-clamp-2">
                                {companyMemory ? companyMemory.split('\n').filter(l => l.trim()).slice(0, 2).join(' | ') : 'Empty'}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Metrics Footer */}
            <div className="p-3 border-t border-border">
                <div className="grid grid-cols-3 gap-1.5">
                    <div className="p-2 bg-zinc-900/30 rounded-md border border-zinc-800/30 text-center">
                        <div className="text-[8px] text-zinc-600 font-mono">TOKENS</div>
                        <div className="text-[11px] font-mono text-zinc-300">{tokensUsed.toLocaleString()}</div>
                    </div>
                    <div className="p-2 bg-zinc-900/30 rounded-md border border-zinc-800/30 text-center">
                        <div className="text-[8px] text-zinc-600 font-mono">MSGS</div>
                        <div className="text-[11px] font-mono text-zinc-300">{messages.length}</div>
                    </div>
                    <div className="p-2 bg-zinc-900/30 rounded-md border border-zinc-800/30 text-center">
                        <div className="text-[8px] text-zinc-600 font-mono">MEMORY</div>
                        <div className="text-[11px] font-mono text-zinc-300">{memoryLines}</div>
                    </div>
                </div>
            </div>
        </div>
    )
}
