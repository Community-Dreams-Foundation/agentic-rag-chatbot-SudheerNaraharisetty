"use client"

import * as React from "react"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useAppStore } from "@/lib/store"
import { Code2, FileSearch, TerminalSquare } from "lucide-react"

export function InspectorPanel() {
    const { activeTab, setActiveTab, traceLogs } = useAppStore()

    // Auto-scroll traces
    const traceEndRef = React.useRef<HTMLDivElement>(null)
    React.useEffect(() => {
        traceEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [traceLogs])

    return (
        <div className="flex flex-col h-full bg-zinc-950 border-l border-zinc-800">
            <Tabs
                value={activeTab}
                onValueChange={(v) => setActiveTab(v as any)}
                className="flex flex-col h-full"
            >
                <div className="p-2 border-b border-zinc-900 bg-zinc-950">
                    <TabsList className="w-full bg-zinc-900 grid grid-cols-3">
                        <TabsTrigger value="trace" className="text-[10px] uppercase font-mono data-[state=active]:bg-zinc-800 data-[state=active]:text-amber-500">
                            <TerminalSquare className="w-3 h-3 mr-2" />
                            Trace
                        </TabsTrigger>
                        <TabsTrigger value="source" className="text-[10px] uppercase font-mono data-[state=active]:bg-zinc-800 data-[state=active]:text-amber-500">
                            <FileSearch className="w-3 h-3 mr-2" />
                            Source
                        </TabsTrigger>
                        <TabsTrigger value="visual" className="text-[10px] uppercase font-mono data-[state=active]:bg-zinc-800 data-[state=active]:text-amber-500">
                            <Code2 className="w-3 h-3 mr-2" />
                            Visual
                        </TabsTrigger>
                    </TabsList>
                </div>

                <div className="flex-1 overflow-hidden relative">
                    {/* TRACE TAB */}
                    <TabsContent value="trace" className="h-full m-0 p-0 absolute inset-0">
                        <ScrollArea className="h-full">
                            <div className="p-4 font-mono text-[10px] space-y-1">
                                {traceLogs.length === 0 && (
                                    <div className="text-zinc-700 italic text-center mt-10">No agent traces yet.</div>
                                )}
                                {traceLogs.map((log, i) => (
                                    <div key={i} className="text-zinc-500 border-l-2 border-zinc-800 pl-2 py-1">
                                        {log}
                                    </div>
                                ))}
                                <div ref={traceEndRef} />
                            </div>
                        </ScrollArea>
                    </TabsContent>

                    {/* SOURCE TAB */}
                    <TabsContent value="source" className="h-full m-0 p-4 absolute inset-0">
                        <div className="flex items-center justify-center h-full text-zinc-700 text-xs italic border-2 border-dashed border-zinc-900 rounded">
                            Hover over citations [1] to view source context.
                        </div>
                    </TabsContent>

                    {/* VISUAL TAB */}
                    <TabsContent value="visual" className="h-full m-0 p-4 absolute inset-0">
                        <div className="flex items-center justify-center h-full text-zinc-700 text-xs italic border-2 border-dashed border-zinc-900 rounded">
                            Tool outputs (Weather maps, Charts) will appear here.
                        </div>
                    </TabsContent>
                </div>
            </Tabs>
        </div>
    )
}
