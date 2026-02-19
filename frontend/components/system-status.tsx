"use client"

import * as React from "react"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { useAppStore } from "@/lib/store"
import { Activity, Database, Server, UploadCloud, Loader2, Cpu } from "lucide-react"
import { apiClient } from "@/lib/api"
import { Button } from "@/components/ui/button"

export function SystemStatus() {
    const { isConnected, setIsConnected, files, addFile, tokensUsed, model, setModel, messages } = useAppStore()

    React.useEffect(() => {
        apiClient.checkHealth().then((data) => {
            if (data.status === "healthy") {
                setIsConnected(true)
            }
        })

        const interval = setInterval(() => {
            apiClient.checkHealth().then((data) => {
                setIsConnected(data.status === "healthy")
            })
        }, 10000)

        return () => clearInterval(interval)
    }, [setIsConnected])

    const [isUploading, setIsUploading] = React.useState(false)

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            setIsUploading(true);
            try {
                const res = await apiClient.uploadFile(file);
                if (res.status === 'success') {
                    addFile(res.filename || file.name)
                }
            } catch (error) {
                console.error("Upload failed", error)
            } finally {
                setIsUploading(false);
            }
        }
    }

    return (
        <div className="flex flex-col h-full bg-zinc-950 border-r border-zinc-800">
            {/* Header */}
            <div className="p-4 border-b border-zinc-900">
                <h2 className="text-xs font-bold tracking-widest text-zinc-500 uppercase flex items-center gap-2">
                    <Server className="w-4 h-4" />
                    System Context
                </h2>
                <div className="mt-2 flex items-center justify-between">
                    <span className="text-sm text-zinc-400 font-mono">AGENT_V1</span>
                    <div className="flex items-center gap-2">
                        <span className="text-[10px] text-zinc-600">STATUS</span>
                        <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]' : 'bg-red-500'}`} />
                    </div>
                </div>
            </div>

            {/* Model Selection */}
            <div className="p-3 border-b border-zinc-900">
                <h3 className="text-[10px] uppercase tracking-wider text-zinc-600 mb-2 flex items-center gap-1">
                    <Cpu className="w-3 h-3" />
                    Model
                </h3>
                <div className="flex gap-1">
                    <button
                        onClick={() => setModel('openrouter')}
                        className={`flex-1 text-[10px] font-mono py-1.5 px-2 rounded border transition-colors ${
                            model === 'openrouter'
                                ? 'border-amber-500/50 bg-amber-500/10 text-amber-400'
                                : 'border-zinc-800 text-zinc-600 hover:border-zinc-700'
                        }`}
                    >
                        Llama 3.3 70B
                    </button>
                    <button
                        onClick={() => setModel('groq')}
                        className={`flex-1 text-[10px] font-mono py-1.5 px-2 rounded border transition-colors ${
                            model === 'groq'
                                ? 'border-amber-500/50 bg-amber-500/10 text-amber-400'
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
                        className="w-full border-dashed border-zinc-700 hover:border-amber-500/50 hover:bg-zinc-900 text-zinc-400"
                        onClick={() => document.getElementById('file-upload')?.click()}
                        disabled={isUploading}
                    >
                        {isUploading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <UploadCloud className="w-4 h-4 mr-2" />}
                        {isUploading ? "Indexing..." : "Upload Document"}
                    </Button>
                    <input
                        id="file-upload"
                        type="file"
                        className="hidden"
                        onChange={handleUpload}
                        accept=".pdf,.txt,.md,.html"
                    />
                </div>

                <Separator className="bg-zinc-900" />

                <div className="p-2">
                    <h3 className="text-[10px] items-center uppercase tracking-wider text-zinc-600 mb-2 px-2 flex gap-1">
                        <Database className="w-3 h-3" />
                        Active Knowledge
                    </h3>
                    <ScrollArea className="h-[200px]">
                        {files.length === 0 ? (
                            <div className="px-4 py-8 text-center text-zinc-700 text-xs italic font-mono">
                                No documents indexed.
                            </div>
                        ) : (
                            <div className="space-y-1 px-2">
                                {files.map((f, i) => (
                                    <div key={i} className="flex items-center justify-between p-2 rounded bg-zinc-900/50 border border-zinc-800/50">
                                        <span className="text-xs text-zinc-300 truncate max-w-[120px] font-mono">{f}</span>
                                        <Badge variant="secondary" className="text-[10px] h-4 bg-zinc-800 text-zinc-500">READY</Badge>
                                    </div>
                                ))}
                            </div>
                        )}
                    </ScrollArea>
                </div>
            </div>

            {/* Metrics Footer */}
            <div className="p-4 border-t border-zinc-900 bg-zinc-950/50">
                <h3 className="text-[10px] uppercase tracking-wider text-zinc-600 mb-3 flex items-center gap-1">
                    <Activity className="w-3 h-3 text-amber-600" />
                    Session
                </h3>
                <div className="grid grid-cols-2 gap-2">
                    <div className="p-2 bg-zinc-900 rounded border border-zinc-800">
                        <div className="text-[10px] text-zinc-500">TOKENS</div>
                        <div className="text-sm font-mono text-zinc-200">{tokensUsed.toLocaleString()}</div>
                    </div>
                    <div className="p-2 bg-zinc-900 rounded border border-zinc-800">
                        <div className="text-[10px] text-zinc-500">MESSAGES</div>
                        <div className="text-sm font-mono text-zinc-200">{messages.length}</div>
                    </div>
                </div>
            </div>
        </div>
    )
}
