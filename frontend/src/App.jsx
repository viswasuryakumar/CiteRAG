// Day 12: the CiteRAG chat page.
// It opens a Server-Sent Events stream to GET /ask?q=... and shows the answer as it arrives:
//   "sources" event -> the 5 excerpts found (so [n] markers can link to them right away)
//   "token" events  -> pieces of the answer, appended as they come
//   "done" event    -> the checked citations; only the sources actually cited are listed
import { useRef, useState } from 'react'

// Where the API runs. On Day 13 this is set to the AWS address at build time (VITE_API_URL).
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const EXAMPLES = [
  'How do I return a 404 error?',
  'How does OAuth2PasswordBearer get the token from the request?',
  'How do I run something after the response is sent?',
  "How do I enable FastAPI's built-in admin panel?",
]

export default function App() {
  const [question, setQuestion] = useState('')
  const [asked, setAsked] = useState('')       // the question being answered
  const [answer, setAnswer] = useState('')     // grows token by token
  const [excerpts, setExcerpts] = useState([]) // the 5 retrieved excerpts
  const [result, setResult] = useState(null)   // the final "done" data
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const streamRef = useRef(null)               // the open EventSource, so we can close it

  function ask(text) {
    const q = text.trim()
    if (q.length < 3 || loading) return

    streamRef.current?.close()   // stop any previous answer still streaming
    setAsked(q); setAnswer(''); setExcerpts([]); setResult(null); setError(''); setLoading(true)

    // EventSource is built into the browser: it opens the stream and calls our
    // listeners for each named event. Every event's data is JSON.
    const stream = new EventSource(`${API_URL}/ask?q=${encodeURIComponent(q)}`)
    streamRef.current = stream

    stream.addEventListener('sources', (e) => setExcerpts(JSON.parse(e.data)))
    stream.addEventListener('token', (e) => setAnswer((prev) => prev + JSON.parse(e.data)))
    stream.addEventListener('done', (e) => {
      setResult(JSON.parse(e.data))
      setLoading(false)
      stream.close()   // otherwise EventSource would reconnect and ask again
    })
    stream.onerror = () => {
      stream.close()
      setLoading(false)
      // EventSource can't tell us the HTTP status, so this covers both "API down" and
      // "rate limit reached" (429).
      setError('Could not get an answer. The API may be starting up or busy, or you have reached ' +
        'the limit of questions per hour. Please try again in a moment.')
    }
  }

  function onSubmit(e) {
    e.preventDefault()   // stop the browser from reloading the page
    ask(question)
  }

  return (
    <main className="page">
      <header>
        <h1>CiteRAG</h1>
        <p className="tagline">
          Ask about FastAPI. Every answer cites the exact lines of the docs or source code it came from.
        </p>
      </header>

      <form className="ask" onSubmit={onSubmit}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. How do I return a 404 error?"
          maxLength={500}
          aria-label="Your question"
          autoFocus
        />
        <button type="submit" disabled={loading || question.trim().length < 3}>
          {loading ? 'Answering…' : 'Ask'}
        </button>
      </form>

      {!asked && (
        <div className="examples">
          <span>Try:</span>
          {EXAMPLES.map((ex) => (
            <button key={ex} type="button" onClick={() => { setQuestion(ex); ask(ex) }}>
              {ex}
            </button>
          ))}
        </div>
      )}

      {asked && (
        <section className="result">
          <h2 className="question">{asked}</h2>

          {loading && !answer && (
            <p className="status">
              {excerpts.length ? `Found ${excerpts.length} excerpts, writing the answer…` : 'Searching the docs and code…'}
            </p>
          )}

          {error && <p className="error">{error}</p>}

          {result?.abstained ? (
            <p className="abstained">
              I don't know: the FastAPI docs and code I searched don't answer this.
            </p>
          ) : (
            answer && <Answer text={answer} excerpts={excerpts} />
          )}

          {result && result.sources.length > 0 && (
            <div className="sources">
              <h3>Sources</h3>
              <ol>
                {result.sources.map((s) => (
                  <li key={s.n} id={`source-${s.n}`}>
                    <span className="num">[{s.n}]</span>
                    <a href={s.url} target="_blank" rel="noreferrer">
                      {s.file}
                    </a>
                    <span className="lines">lines {s.start_line}–{s.end_line}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {result?.invalid_citations.length > 0 && (
            <p className="warning">Note: the answer cited excerpts that don't exist: {result.invalid_citations.join(', ')}</p>
          )}
        </section>
      )}
    </main>
  )
}

// Shows the answer text: ```code blocks```, `inline code`, and [n] markers as links to GitHub.
function Answer({ text, excerpts }) {
  const parts = text.split('```')   // even parts = normal text, odd parts = code blocks
  return (
    <div className="answer">
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          // Drop the language name on the first line, e.g. "python".
          <pre key={i}><code>{part.replace(/^[a-z]*\n/, '')}</code></pre>
        ) : (
          <Prose key={i} text={part} excerpts={excerpts} />
        ),
      )}
    </div>
  )
}

function Prose({ text, excerpts }) {
  // Split on `code` and [n]; the capturing group keeps the separators in the result.
  const pieces = text.split(/(`[^`]+`|\[\d+\])/)
  return (
    <p>
      {pieces.map((piece, i) => {
        if (piece.startsWith('`') && piece.endsWith('`') && piece.length > 1) {
          return <code key={i}>{piece.slice(1, -1)}</code>
        }
        const marker = piece.match(/^\[(\d+)\]$/)
        if (marker) {
          const excerpt = excerpts.find((x) => x.n === Number(marker[1]))
          return excerpt ? (
            <a key={i} className="cite" href={excerpt.url} target="_blank" rel="noreferrer"
               title={`${excerpt.file}, lines ${excerpt.start_line}–${excerpt.end_line}`}>
              {piece}
            </a>
          ) : <span key={i}>{piece}</span>
        }
        return <span key={i}>{piece}</span>
      })}
    </p>
  )
}
