"use client"

import * as React from "react"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useAppStore, type Citation } from "@/lib/store"
import { apiClient } from "@/lib/api"
import {
    FileSearch, TerminalSquare, Brain, Cloud, Code2,
    Send, Loader2, Play, ChevronDown, ChevronRight
} from "lucide-react"
import { Input } from "@/components/ui/input"

function SourceViewer() {
    const { activeSource, activeCitations, files } = useAppStore()

    // Gather all citations from messages — use raw state + useMemo to avoid
    // infinite re-render (Zustand selectors must return referentially stable values)
    const messages = useAppStore(state => state.messages)
    const allCitations = React.useMemo(
        () => messages.flatMap(m => m.citations || []),
        [messages]
    )

    const displayCitations = activeCitations.length > 0 ? activeCitations : allCitations

    if (!displayCitations.length && !files.length) {
        return (
            <div className="flex flex-col items-center justify-center h-full text-zinc-600 text-xs font-mono p-6">
                <FileSearch className="w-8 h-8 mb-3 text-zinc-700" />
                <p className="text-center leading-relaxed">Upload a document and ask questions to see sources here.</p>
            </div>
        )
    }

    // Group citations by source
    const bySource: Record<string, Citation[]> = {}
    for (const c of displayCitations) {
        if (!bySource[c.source]) bySource[c.source] = []
        bySource[c.source].push(c)
    }

    return (
        <ScrollArea className="h-full">
            <div className="p-4 space-y-4">
                {/* PDF Viewer (if file is a PDF) */}
                {activeSource && activeSource.toLowerCase().endsWith('.pdf') && (
                    <div className="border border-orange-500/20 rounded-lg overflow-hidden">
                        <div className="bg-orange-500/10 px-3 py-2 text-[11px] font-mono text-orange-400 flex items-center gap-1.5">
                            <FileSearch className="w-3.5 h-3.5" />
                            {activeSource}
                        </div>
                        <iframe
                            src={apiClient.getFileUrl(activeSource)}
                            className="w-full h-[300px] bg-zinc-900"
                            title={`PDF Viewer: ${activeSource}`}
                        />
                    </div>
                )}

                {/* Cited Passages */}
                {Object.entries(bySource).map(([source, cits]) => (
                    <div key={source} className="space-y-2">
                        <div className="text-[11px] font-mono text-orange-400/70 uppercase tracking-wider flex items-center gap-1.5">
                            <FileSearch className="w-3.5 h-3.5" />
                            {source}
                        </div>
                        {cits.map((c, i) => (
                            <div key={i} className="citation-highlight text-xs font-mono rounded-md">
                                <span className="text-yellow-400 font-semibold">[{c.locator}]</span>
                                <p className="text-zinc-400 mt-1 leading-relaxed">{c.snippet}</p>
                            </div>
                        ))}
                    </div>
                ))}
            </div>
        </ScrollArea>
    )
}

function MemoryViewer() {
    const { userMemory, companyMemory } = useAppStore()
    const [openUser, setOpenUser] = React.useState(true)
    const [openCompany, setOpenCompany] = React.useState(true)

    const userHasContent = userMemory.trim().length > 50
    const companyHasContent = companyMemory.trim().length > 50

    return (
        <ScrollArea className="h-full">
            <div className="p-4 space-y-4">
                {/* Memory Status Banner */}
                <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-[11px] font-mono ${
                    userHasContent || companyHasContent
                        ? 'bg-green-500/10 border border-green-500/20 text-green-400'
                        : 'bg-zinc-800/50 border border-zinc-700/30 text-zinc-500'
                }`}>
                    <Brain className="w-3.5 h-3.5 shrink-0" />
                    {userHasContent || companyHasContent
                        ? `Persistent memory active — ${userHasContent ? 'user' : ''}${userHasContent && companyHasContent ? ' + ' : ''}${companyHasContent ? 'company' : ''} loaded`
                        : 'No memory stored yet. Chat to build memory.'}
                </div>

                {/* User Memory */}
                <div className={`border rounded-lg overflow-hidden ${userHasContent ? 'border-green-500/20' : 'border-orange-500/15'}`}>
                    <button
                        onClick={() => setOpenUser(!openUser)}
                        className={`flex items-center gap-2 w-full px-3 py-2.5 text-xs font-mono transition-colors ${
                            userHasContent
                                ? 'bg-green-500/5 text-green-400 hover:bg-green-500/10'
                                : 'bg-orange-500/5 text-orange-400 hover:bg-orange-500/10'
                        }`}
                    >
                        {openUser ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                        <Brain className="w-3.5 h-3.5" />
                        USER_MEMORY.md
                        {userHasContent && (
                            <span className="ml-auto text-[10px] bg-green-500/15 text-green-400 px-1.5 py-0.5 rounded">active</span>
                        )}
                    </button>
                    {openUser && (
                        <div className="p-3">
                            {userMemory ? (
                                <pre className="text-xs font-mono text-zinc-400 whitespace-pre-wrap leading-relaxed">{userMemory}</pre>
                            ) : (
                                <p className="text-xs font-mono text-zinc-600 italic">No user memory yet. Chat to build memory.</p>
                            )}
                        </div>
                    )}
                </div>

                {/* Company Memory */}
                <div className={`border rounded-lg overflow-hidden ${companyHasContent ? 'border-green-500/20' : 'border-orange-500/15'}`}>
                    <button
                        onClick={() => setOpenCompany(!openCompany)}
                        className={`flex items-center gap-2 w-full px-3 py-2.5 text-xs font-mono transition-colors ${
                            companyHasContent
                                ? 'bg-green-500/5 text-green-400 hover:bg-green-500/10'
                                : 'bg-orange-500/5 text-orange-400 hover:bg-orange-500/10'
                        }`}
                    >
                        {openCompany ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                        <Brain className="w-3.5 h-3.5" />
                        COMPANY_MEMORY.md
                        {companyHasContent && (
                            <span className="ml-auto text-[10px] bg-green-500/15 text-green-400 px-1.5 py-0.5 rounded">active</span>
                        )}
                    </button>
                    {openCompany && (
                        <div className="p-3">
                            {companyMemory ? (
                                <pre className="text-xs font-mono text-zinc-400 whitespace-pre-wrap leading-relaxed">{companyMemory}</pre>
                            ) : (
                                <p className="text-xs font-mono text-zinc-600 italic">No company memory yet. Share organizational info to build memory.</p>
                            )}
                        </div>
                    )}
                </div>

                <div className="text-[10px] font-mono text-zinc-700 text-center pt-2">
                    Memory is built from conversations via LLM-based selective write
                </div>
            </div>
        </ScrollArea>
    )
}

function ToolsPanel() {
    const [weatherQuery, setWeatherQuery] = React.useState("")
    const [weatherResult, setWeatherResult] = React.useState<any>(null)
    const [weatherLoading, setWeatherLoading] = React.useState(false)

    const [sandboxCode, setSandboxCode] = React.useState("")
    const [sandboxResult, setSandboxResult] = React.useState<any>(null)
    const [sandboxLoading, setSandboxLoading] = React.useState(false)

    const handleWeather = async () => {
        if (!weatherQuery.trim()) return
        setWeatherLoading(true)
        setWeatherResult(null)
        try {
            const result = await apiClient.runWeather(weatherQuery)
            setWeatherResult(result)
        } catch (e) {
            setWeatherResult({ error: String(e) })
        }
        setWeatherLoading(false)
    }

    const handleSandbox = async () => {
        if (!sandboxCode.trim()) return
        setSandboxLoading(true)
        setSandboxResult(null)
        try {
            const result = await apiClient.runSandbox(sandboxCode)
            setSandboxResult(result)
        } catch (e) {
            setSandboxResult({ error: String(e) })
        }
        setSandboxLoading(false)
    }

    return (
        <ScrollArea className="h-full">
            <div className="p-4 space-y-6">
                {/* Weather Tool */}
                <div className="space-y-3">
                    <div className="text-[11px] font-mono text-orange-400/70 uppercase tracking-wider flex items-center gap-1.5">
                        <Cloud className="w-3.5 h-3.5" />
                        Weather (OpenMeteo)
                    </div>
                    <div className="flex gap-2">
                        <Input
                            value={weatherQuery}
                            onChange={e => setWeatherQuery(e.target.value)}
                            placeholder="e.g. weather in Tokyo"
                            className="text-xs font-mono bg-zinc-900/50 border-zinc-800 h-9"
                            onKeyDown={e => e.key === 'Enter' && handleWeather()}
                        />
                        <button
                            onClick={handleWeather}
                            disabled={weatherLoading}
                            className="px-3 h-9 bg-orange-500/10 text-orange-400 border border-orange-500/20 rounded-md text-xs font-mono hover:bg-orange-500/20 transition-colors disabled:opacity-50 shrink-0"
                        >
                            {weatherLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
                        </button>
                    </div>
                    {weatherResult && (
                        <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-3">
                            <pre className="text-[11px] font-mono text-zinc-400 whitespace-pre-wrap overflow-auto max-h-[200px]">
                                {JSON.stringify(weatherResult, null, 2)}
                            </pre>
                        </div>
                    )}
                </div>

                {/* Sandbox Tool */}
                <div className="space-y-3">
                    <div className="text-[11px] font-mono text-orange-400/70 uppercase tracking-wider flex items-center gap-1.5">
                        <Code2 className="w-3.5 h-3.5" />
                        Python Sandbox
                    </div>
                    <div className="relative">
                        <textarea
                            value={sandboxCode}
                            onChange={e => setSandboxCode(e.target.value)}
                            placeholder={"# Safe Python execution\nimport math\nprint(math.pi)"}
                            className="w-full h-28 text-xs font-mono bg-zinc-900/50 border border-zinc-800 rounded-lg p-3 text-zinc-300 placeholder:text-zinc-700 resize-none focus:outline-none focus:border-orange-500/30"
                        />
                    </div>
                    <button
                        onClick={handleSandbox}
                        disabled={sandboxLoading || !sandboxCode.trim()}
                        className="flex items-center gap-1.5 px-3 py-2 bg-orange-500/10 text-orange-400 border border-orange-500/20 rounded-lg text-xs font-mono hover:bg-orange-500/20 transition-colors disabled:opacity-50"
                    >
                        {sandboxLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                        Execute
                    </button>
                    {sandboxResult && (
                        <div className={`border rounded-lg p-3 ${
                            sandboxResult.success
                                ? 'bg-green-500/5 border-green-500/20'
                                : 'bg-red-500/5 border-red-500/20'
                        }`}>
                            <div className="text-[10px] font-mono text-zinc-500 mb-1.5 uppercase">
                                {sandboxResult.success ? 'Success' : 'Error'}
                                {sandboxResult.execution_time && ` (${sandboxResult.execution_time}s)`}
                            </div>
                            <pre className="text-xs font-mono whitespace-pre-wrap overflow-auto max-h-[200px]">
                                <span className={sandboxResult.success ? 'text-green-400' : 'text-red-400'}>
                                    {sandboxResult.output || sandboxResult.error || sandboxResult.result || 'No output'}
                                </span>
                            </pre>
                        </div>
                    )}
                </div>
            </div>
        </ScrollArea>
    )
}

export function InspectorPanel() {
    const { activeTab, setActiveTab, traceLogs } = useAppStore()

    const traceEndRef = React.useRef<HTMLDivElement>(null)
    React.useEffect(() => {
        traceEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [traceLogs])

    return (
        <div className="flex flex-col h-full bg-background border-l border-border">
            <Tabs
                value={activeTab}
                onValueChange={(v) => setActiveTab(v as any)}
                className="flex flex-col h-full"
            >
                <div className="p-2 border-b border-border bg-background">
                    <TabsList className="w-full bg-zinc-900/50 grid grid-cols-4">
                        <TabsTrigger value="trace" className="text-[10px] uppercase font-mono data-[state=active]:bg-orange-500/10 data-[state=active]:text-orange-400">
                            <TerminalSquare className="w-3.5 h-3.5 mr-1" />
                            Trace
                        </TabsTrigger>
                        <TabsTrigger value="source" className="text-[10px] uppercase font-mono data-[state=active]:bg-orange-500/10 data-[state=active]:text-orange-400">
                            <FileSearch className="w-3.5 h-3.5 mr-1" />
                            Source
                        </TabsTrigger>
                        <TabsTrigger value="memory" className="text-[10px] uppercase font-mono data-[state=active]:bg-orange-500/10 data-[state=active]:text-orange-400">
                            <Brain className="w-3.5 h-3.5 mr-1" />
                            Memory
                        </TabsTrigger>
                        <TabsTrigger value="tools" className="text-[10px] uppercase font-mono data-[state=active]:bg-orange-500/10 data-[state=active]:text-orange-400">
                            <Code2 className="w-3.5 h-3.5 mr-1" />
                            Tools
                        </TabsTrigger>
                    </TabsList>
                </div>

                <div className="flex-1 overflow-hidden relative">
                    {/* TRACE TAB */}
                    <TabsContent value="trace" className="h-full m-0 p-0 absolute inset-0">
                        <ScrollArea className="h-full">
                            <div className="p-4 font-mono text-[11px] space-y-1">
                                {traceLogs.length === 0 && (
                                    <div className="flex flex-col items-center justify-center h-40 text-zinc-700">
                                        <TerminalSquare className="w-6 h-6 mb-2" />
                                        <span className="italic text-xs">No agent traces yet.</span>
                                    </div>
                                )}
                                {traceLogs.map((log, i) => (
                                    <div key={i} className={`py-1 pl-2 border-l-2 ${
                                        log.includes('Tool:') ? 'border-orange-500/40 text-orange-400/70' :
                                        log.includes('Error') ? 'border-red-500/40 text-red-400/70' :
                                        log.includes('Citations') ? 'border-yellow-500/40 text-yellow-400/70' :
                                        log.includes('complete') ? 'border-green-500/40 text-green-400/70' :
                                        'border-zinc-800 text-zinc-500'
                                    }`}>
                                        {log}
                                    </div>
                                ))}
                                <div ref={traceEndRef} />
                            </div>
                        </ScrollArea>
                    </TabsContent>

                    {/* SOURCE TAB */}
                    <TabsContent value="source" className="h-full m-0 p-0 absolute inset-0">
                        <SourceViewer />
                    </TabsContent>

                    {/* MEMORY TAB */}
                    <TabsContent value="memory" className="h-full m-0 p-0 absolute inset-0">
                        <MemoryViewer />
                    </TabsContent>

                    {/* TOOLS TAB */}
                    <TabsContent value="tools" className="h-full m-0 p-0 absolute inset-0">
                        <ToolsPanel />
                    </TabsContent>
                </div>
            </Tabs>
        </div>
    )
}
