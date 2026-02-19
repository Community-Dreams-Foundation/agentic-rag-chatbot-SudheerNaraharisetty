"use client"

import * as React from "react"
import { ChevronDown, BrainCircuit } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

interface ThinkingBlockProps {
    content: string
    isStreaming?: boolean
}

export function ThinkingBlock({ content, isStreaming }: ThinkingBlockProps) {
    const [isOpen, setIsOpen] = React.useState(true) // Default open while streaming? Or default closed? User said "when answer ready thinking becomes openable dropdown". 
    // So: Open while streaming (thinking), then collapses when done? Or always collapsible?
    // Let's make it always collapsible, default open if it's the last message and streaming?

    // Auto-collapse when streaming finishes could be nice, but let's stick to manual.

    if (!content) return null

    return (
        <div className="my-2 border border-amber-500/20 rounded bg-amber-500/5 overflow-hidden">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center gap-2 p-2 px-3 text-xs font-mono text-amber-600/80 hover:bg-amber-500/10 transition-colors"
            >
                <BrainCircuit className="w-3 h-3" />
                <span className="uppercase tracking-wider font-bold">Thought Process</span>
                {isStreaming && <span className="animate-pulse ml-1">...</span>}
                <ChevronDown className={`w-3 h-3 ml-auto transition-transform ${isOpen ? 'rotate-180' : ''}`} />
            </button>
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden"
                    >
                        <div className="p-3 pt-0 text-xs font-mono text-amber-700/80 whitespace-pre-wrap leading-relaxed border-t border-amber-500/10 dashed">
                            {content}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}
