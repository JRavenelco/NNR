import React, { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

// Inicializar Mermaid una sola vez
mermaid.initialize({
  startOnLoad: false, // No queremos que renderice automáticamente todos los div.mermaid
  theme: 'default', // Puedes cambiar el tema: 'default', 'forest', 'dark', 'neutral'
  // securityLevel: 'loose', // Descomentar si hay problemas con scripts/eventos en los diagramas
});

const MermaidDiagram = ({ chart }) => {
  const mermaidRef = useRef(null);

  useEffect(() => {
    if (mermaidRef.current && chart) {
      mermaid.render(
        `mermaid-graph-${Math.random().toString(36).substring(7)}`, // ID único para el SVG
        chart,
        (svgCode) => {
          if (mermaidRef.current) {
            mermaidRef.current.innerHTML = svgCode;
          }
        },
        mermaidRef.current
      );
    }
  }, [chart]);

  // Si no hay 'chart', no renderizar nada o un placeholder
  if (!chart) {
    return <p>Cargando diagrama...</p>;
  }

  return <div ref={mermaidRef} className="mermaid-diagram-container"></div>;
};

export default MermaidDiagram;
