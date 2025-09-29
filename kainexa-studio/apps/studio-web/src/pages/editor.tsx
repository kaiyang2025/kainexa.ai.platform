// pages/editor.tsx
import dynamic from 'next/dynamic';

// ReactFlow는 브라우저 전용이라 SSR 끄기
const WorkflowEditor = dynamic(
  () => import('@/components/workflow-editor/WorkflowEditor'),
  { ssr: false }
);

export default function EditorPage() {
  return <WorkflowEditor />;
}
