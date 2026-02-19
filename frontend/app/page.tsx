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
    <main className="h-screen w-screen bg-black overflow-hidden flex flex-col">
      <ResizablePanelGroup direction="horizontal" className="flex-1">
        {/* Left Sidebar: System Context */}
        <ResizablePanel defaultSize={20} minSize={15} maxSize={25} className="min-w-[250px]">
          <SystemStatus />
        </ResizablePanel>

        <ResizableHandle className="bg-zinc-800" />

        {/* Middle Pane: Chat Interface */}
        <ResizablePanel defaultSize={50} minSize={30}>
          <ChatInterface />
        </ResizablePanel>

        <ResizableHandle className="bg-zinc-800" />

        {/* Right Sidebar: Inspector */}
        <ResizablePanel defaultSize={30} minSize={20} className="min-w-[300px]">
          <InspectorPanel />
        </ResizablePanel>
      </ResizablePanelGroup>
    </main>
  );
}
