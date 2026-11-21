import { useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [llmResponse, setLlmResponse] = useState('');
  const [loading, setLoading] = useState(false);

  // Survives re-renders (the old `let` was reset on every setState)
  const progressRef = useRef('');

  const handleInputChange = (e) => setQuestion(e.target.value);

  const fetchData = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setLlmResponse('Please be patient – the first call can take a while while the model loads…');
    progressRef.current = '';

    try {
      const response = await fetch('http://localhost:8000/', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question })
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || `HTTP ${response.status}`);
      }

      const reader = response.body
        .pipeThrough(new TextDecoderStream())
        .getReader();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        progressRef.current += value || '';
        setLlmResponse(progressRef.current);

        // Auto-scroll the response area
        const el = document.querySelector('.llm-response');
        if (el) {
          el.scrollTop = el.scrollHeight;
        }
      }
    } catch (err) {
      console.error(err);
      setLlmResponse('Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <h1>The Developer's Guide to AI</h1>
      <div className="card">
        <textarea
          value={question}
          onChange={handleInputChange}
          placeholder="Ask me a question!"
          rows={5}
          style={{ width: '100%', fontSize: '1.1em', padding: '0.6em', boxSizing: 'border-box' }}
        />
        <br />
        <button onClick={fetchData} disabled={loading}>
          {loading ? 'Thinking…' : 'Ask AI'}
        </button>

        <div
          className="llm-response"
          style={{
            maxHeight: '60vh',
            overflowY: 'auto',
            textAlign: 'left',
            marginTop: '1.5rem',
            padding: '1rem',
            background: '#f5f5f5',
            borderRadius: '8px'
          }}
        >
          <ReactMarkdown>{llmResponse}</ReactMarkdown>
        </div>
      </div>
    </>
  );
}

export default App;
