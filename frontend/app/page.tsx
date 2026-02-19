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
    <main className="h-screen w-screen bg-background overflow-hidden flex">
      {/* Fixed Left Sidebar */}
      <div className="w-[280px] shrink-0">
        <SystemStatus />
      </div>

      {/* Resizable Center + Right */}
      <ResizablePanelGroup orientation="horizontal" className="flex-1">
        <ResizablePanel defaultSize={62} minSize={40}>
          <ChatInterface />
        </ResizablePanel>

        <ResizableHandle className="bg-border hover:bg-orange-500/20 transition-colors" />

        <ResizablePanel defaultSize={38} minSize={25}>
          <InspectorPanel />
        </ResizablePanel>
      </ResizablePanelGroup>
    </main>
  );
}
