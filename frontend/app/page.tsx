"use client"

import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable"
import { SystemStatus } from "@/components/system-status"
import { ChatInterface } from "@/components/chat-interface"
import { InspectorPanel } from "@/components/inspector-panel"

export default function Home() {
  return (
    <main className="h-screen w-screen bg-background overflow-hidden flex flex-col">
      <ResizablePanelGroup orientation="horizontal" className="flex-1">
        {/* Left Sidebar: System Context */}
        <ResizablePanel defaultSize={18} minSize={14} maxSize={24} className="min-w-[220px]">
          <SystemStatus />
        </ResizablePanel>

        <ResizableHandle className="bg-border hover:bg-orange-500/20 transition-colors" />

        {/* Middle Pane: Chat Interface */}
        <ResizablePanel defaultSize={50} minSize={30}>
          <ChatInterface />
        </ResizablePanel>

        <ResizableHandle className="bg-border hover:bg-orange-500/20 transition-colors" />

        {/* Right Sidebar: Inspector */}
        <ResizablePanel defaultSize={32} minSize={20} className="min-w-[280px]">
          <InspectorPanel />
        </ResizablePanel>
      </ResizablePanelGroup>
    </main>
  );
}
