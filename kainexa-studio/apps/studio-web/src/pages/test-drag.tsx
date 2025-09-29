// apps/studio-web/src/pages/test-drag.tsx
export default function TestDrag() {
  return (
    <div style={{ padding: '50px' }}>
      <h1>드래그 테스트</h1>
      
      <div 
        draggable
        onDragStart={(e) => {
          console.log('Drag started!');
          e.dataTransfer.setData('text', 'hello');
        }}
        style={{
          width: '200px',
          height: '100px',
          backgroundColor: '#3b82f6',
          color: 'white',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'grab',
          borderRadius: '8px',
        }}
      >
        이 박스를 드래그해보세요
      </div>
      
      <div
        onDrop={(e) => {
          e.preventDefault();
          console.log('Dropped!', e.dataTransfer.getData('text'));
        }}
        onDragOver={(e) => e.preventDefault()}
        style={{
          marginTop: '50px',
          width: '400px',
          height: '200px',
          border: '2px dashed #ccc',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        여기에 놓으세요
      </div>
    </div>
  );
}