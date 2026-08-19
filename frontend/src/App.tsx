import React, { useState, useEffect, useRef } from 'react'
import { 
  Brain, 
  Search, 
  Database, 
  Activity, 
  Send, 
  Terminal, 
  User, 
  Plus, 
  AlertTriangle, 
  CheckCircle, 
  FileText, 
  Layers, 
  AlertCircle, 
  CornerDownLeft, 
  Sparkles, 
  RefreshCw
} from 'lucide-react'

// Match types with FastAPI backend
interface MessageItem {
  role: 'user' | 'assistant' | 'system';
  content: string;
  source?: string;
}

interface ChatStep {
  node: string;
  message: string;
  tool_calls?: Array<{
    name: string;
    args: Record<string, any>;
    id?: string;
  }>;
}

interface IngestedDoc {
  id: string;
  text: string;
  source: string;
  metadata?: Record<string, any>;
}

interface SearchResultItem {
  text: string;
  metadata: {
    source?: string;
    [key: string]: any;
  };
  score: number;
}

interface SystemStatus {
  status: string;
  qdrant?: {
    collection_name: string;
    exists: boolean;
    path: string;
  };
  models?: {
    llm: string;
    dense_embeddings: string;
    sparse_embeddings: string;
    reranker: string;
  };
  detail?: string;
}

const BACKEND_URL = 'http://127.0.0.1:8000';

function App() {
  // Tabs: 'chat' | 'ingest' | 'search' | 'status'
  const [activeTab, setActiveTab] = useState<'chat' | 'ingest' | 'search'>('chat');
  
  // Chat State
  const [messages, setMessages] = useState<MessageItem[]>([
    {
      role: 'system',
      content: 'Welcome to Aethelgard. I am your autonomous research assistant, powered by LangGraph, Qdrant Hybrid Search, and a Cross-Encoder Reranking pipeline. Ask me anything grounded in our database.'
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [chatSteps, setChatSteps] = useState<ChatStep[]>([]);
  const [needsHuman, setNeedsHuman] = useState(false);
  const [humanGuidance, setHumanGuidance] = useState('');
  const [chatError, setChatError] = useState<string | null>(null);

  // Ingestion State
  const [ingestText, setIngestText] = useState('');
  const [ingestSource, setIngestSource] = useState('');
  const [ingestTags, setIngestTags] = useState('');
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState<{ type: 'success' | 'error'; msg: string } | null>(null);
  const [ingestedDocs, setIngestedDocs] = useState<IngestedDoc[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);

  // Direct Search Playground State
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResultItem[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchLimit, setSearchLimit] = useState(3);
  const [searchError, setSearchError] = useState<string | null>(null);

  // System Health
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isCheckingStatus, setIsCheckingStatus] = useState(false);

  // Scrolling reference
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Fetch system status on load
  const checkSystemStatus = async () => {
    setIsCheckingStatus(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/status`);
      if (response.ok) {
        const data = await response.json();
        setSystemStatus(data);
        setIsConnected(data.status === 'online');
      } else {
        setIsConnected(false);
      }
    } catch (e) {
      setIsConnected(false);
      console.error("Failed to connect to backend", e);
    } finally {
      setIsCheckingStatus(false);
    }
  };

  // Fetch ingested docs
  const fetchIngestedDocs = async () => {
    setIsLoadingDocs(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/documents`);
      if (response.ok) {
        const data = await response.json();
        setIngestedDocs(data);
      }
    } catch (e) {
      console.error("Failed to fetch ingested documents", e);
    } finally {
      setIsLoadingDocs(false);
    }
  };

  useEffect(() => {
    checkSystemStatus();
    fetchIngestedDocs();
  }, []);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isChatLoading, chatSteps]);

  // Send message to agent
  const handleSendMessage = async (customMessage?: string) => {
    const query = customMessage || inputMessage;
    if (!query.trim()) return;

    if (!customMessage) setInputMessage('');
    setChatError(null);
    setIsChatLoading(true);
    setChatSteps([]);

    // Temporary append of user message in local view before server response
    const newUserMsg: MessageItem = { role: 'user', content: query };
    setMessages(prev => [...prev, newUserMsg]);

    try {
      // Reconstruct the message history (excluding system welcome message)
      const apiHistory = messages
        .filter(m => m.content !== messages[0].content)
        .map(m => ({
          role: m.role,
          content: m.content
        }));

      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: query,
          history: apiHistory
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned code ${response.status}`);
      }

      const data = await response.json();
      
      // Update state from server response
      setMessages(data.history);
      setNeedsHuman(data.needs_human);
      setChatSteps(data.steps);
      
      // If server generated a system escalation message
      if (data.needs_human) {
        setNeedsHuman(true);
      } else {
        setNeedsHuman(false);
      }

    } catch (e: any) {
      setChatError(e.message || "Failed to communicate with LangGraph agent.");
      // Rollback last local message to keep history correct
      setMessages(prev => prev.slice(0, -1));
    } finally {
      setIsChatLoading(false);
    }
  };

  // Submit human guidance override
  const handleResolveEscalation = async () => {
    if (!humanGuidance.trim()) return;
    const guidance = `[HUMAN SUPERVISOR OVERRIDE]: ${humanGuidance}`;
    setHumanGuidance('');
    setNeedsHuman(false);
    handleSendMessage(guidance);
  };

  // Run document ingestion
  const handleIngest = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ingestText.trim() || !ingestSource.trim()) {
      setIngestResult({ type: 'error', msg: 'Document text and source name are required.' });
      return;
    }

    setIsIngesting(true);
    setIngestResult(null);

    // Formulate metadata
    const metadata: Record<string, any> = {};
    if (ingestTags.trim()) {
      metadata.tags = ingestTags.split(',').map(t => t.trim());
    }
    metadata.timestamp = new Date().toISOString();

    try {
      const response = await fetch(`${BACKEND_URL}/api/documents`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: ingestText,
          source: ingestSource,
          metadata
        })
      });

      if (response.ok) {
        setIngestResult({ type: 'success', msg: 'Document embedded and ingested into Qdrant!' });
        setIngestText('');
        setIngestSource('');
        setIngestTags('');
        fetchIngestedDocs(); // Refresh the list
      } else {
        const errData = await response.json();
        setIngestResult({ type: 'error', msg: errData.detail || 'Ingestion failed' });
      }
    } catch (e: any) {
      setIngestResult({ type: 'error', msg: e.message || 'Error occurred during ingestion' });
    } finally {
      setIsIngesting(false);
    }
  };

  // Run direct search
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    setSearchError(null);
    setSearchResults([]);

    try {
      const response = await fetch(`${BACKEND_URL}/api/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: searchQuery,
          limit: searchLimit
        })
      });

      if (response.ok) {
        const data = await response.json();
        setSearchResults(data);
      } else {
        const errData = await response.json();
        setSearchError(errData.detail || 'Search query failed');
      }
    } catch (e: any) {
      setSearchError(e.message || 'Error querying search api');
    } finally {
      setIsSearching(false);
    }
  };

  // Status Badge Helper
  const renderStatusBadge = () => {
    if (isCheckingStatus) {
      return (
        <span className="flex items-center gap-1.5 px-3 py-1 text-xs rounded-full border border-yellow-500/20 bg-yellow-500/10 text-yellow-400">
          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
          Checking Sync...
        </span>
      );
    }
    if (isConnected) {
      return (
        <span className="flex items-center gap-1.5 px-3 py-1 text-xs rounded-full border border-emerald-500/20 bg-emerald-500/10 text-emerald-400">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
          Agent Core Online
        </span>
      );
    }
    return (
      <span className="flex items-center gap-1.5 px-3 py-1 text-xs rounded-full border border-red-500/20 bg-red-500/10 text-red-400">
        <span className="w-1.5 h-1.5 rounded-full bg-red-400"></span>
        Core Offline
      </span>
    );
  };

  return (
    <div className="min-h-screen text-slate-100 font-sans flex flex-col antialiased">
      {/* Top Navbar */}
      <header className="border-b border-white/5 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-600 shadow-md shadow-cyan-500/20">
              <Brain className="w-6 h-6 text-slate-900" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white font-outfit flex items-center gap-2">
                AETHELGARD
                <span className="text-xs font-semibold px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 tracking-wider">AGENTIC RAG</span>
              </h1>
              <p className="text-[10px] text-slate-400 font-mono">Orchestrated with LangGraph & Qdrant</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {renderStatusBadge()}
            <button 
              onClick={checkSystemStatus}
              className="p-1.5 rounded-lg border border-white/10 hover:bg-white/5 transition duration-200 text-slate-400 hover:text-white"
              title="Refresh core connection"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <div className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col lg:flex-row gap-6">
        
        {/* Navigation / Control panel on Left */}
        <aside className="w-full lg:w-64 flex flex-col gap-4 shrink-0">
          <div className="glass-card rounded-2xl p-4 flex flex-col gap-1">
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest px-2 mb-2 font-mono">Navigation</h2>
            
            <button
              onClick={() => setActiveTab('chat')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition duration-200 ${
                activeTab === 'chat' 
                  ? 'bg-gradient-to-r from-cyan-500/10 to-indigo-500/5 text-cyan-400 border-l-2 border-cyan-400 font-semibold' 
                  : 'text-slate-400 hover:text-slate-200 hover:bg-white/5 border-l-2 border-transparent'
              }`}
            >
              <Brain className="w-4.5 h-4.5" />
              Agent Chat
            </button>

            <button
              onClick={() => setActiveTab('ingest')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition duration-200 ${
                activeTab === 'ingest' 
                  ? 'bg-gradient-to-r from-cyan-500/10 to-indigo-500/5 text-cyan-400 border-l-2 border-cyan-400 font-semibold' 
                  : 'text-slate-400 hover:text-slate-200 hover:bg-white/5 border-l-2 border-transparent'
              }`}
            >
              <Database className="w-4.5 h-4.5" />
              Document Store
            </button>

            <button
              onClick={() => setActiveTab('search')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition duration-200 ${
                activeTab === 'search' 
                  ? 'bg-gradient-to-r from-cyan-500/10 to-indigo-500/5 text-cyan-400 border-l-2 border-cyan-400 font-semibold' 
                  : 'text-slate-400 hover:text-slate-200 hover:bg-white/5 border-l-2 border-transparent'
              }`}
            >
              <Search className="w-4.5 h-4.5" />
              Search Sandbox
            </button>
          </div>

          {/* Model Status Card */}
          {systemStatus && (
            <div className="glass-card rounded-2xl p-4 flex flex-col gap-3 text-xs">
              <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest font-mono flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-cyan-400" />
                Pipeline Info
              </h2>
              
              <div className="space-y-2 border-t border-white/5 pt-2 font-mono">
                <div>
                  <span className="text-slate-500">LLM:</span>
                  <p className="text-slate-300 font-medium text-[11px] truncate">{systemStatus.models?.llm}</p>
                </div>
                <div>
                  <span className="text-slate-500">Dense Embedding:</span>
                  <p className="text-slate-300 font-medium text-[11px] truncate">{systemStatus.models?.dense_embeddings}</p>
                </div>
                <div>
                  <span className="text-slate-500">Sparse Embedding:</span>
                  <p className="text-slate-300 font-medium text-[11px] truncate">{systemStatus.models?.sparse_embeddings}</p>
                </div>
                <div>
                  <span className="text-slate-500">Reranker Model:</span>
                  <p className="text-slate-300 font-medium text-[11px] truncate">{systemStatus.models?.reranker}</p>
                </div>
              </div>
            </div>
          )}
        </aside>

        {/* Content Panel on Right */}
        <main className="flex-1 flex flex-col min-w-0">
          
          {/* TAB 1: Chat Interface */}
          {activeTab === 'chat' && (
            <div className="flex-1 flex flex-col glass-card rounded-3xl overflow-hidden h-[calc(100vh-12rem)] min-h-[500px]">
              
              {/* Chat Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((msg, index) => {
                  const isAgent = msg.role === 'assistant';
                  const isSystem = msg.role === 'system';
                  
                  return (
                    <div 
                      key={index} 
                      className={`flex gap-3 max-w-[85%] ${
                        isSystem 
                          ? 'mx-auto w-full max-w-xl text-center justify-center' 
                          : isAgent 
                            ? 'mr-auto items-start' 
                            : 'ml-auto flex-row-reverse items-start'
                      } animate-fade-in-up`}
                    >
                      {/* Avatar */}
                      {!isSystem && (
                        <div className={`p-2 rounded-xl shrink-0 ${
                          isAgent 
                            ? 'bg-slate-900 border border-cyan-500/20 text-cyan-400' 
                            : 'bg-indigo-600 text-white'
                        }`}>
                          {isAgent ? <Brain className="w-4 h-4" /> : <User className="w-4 h-4" />}
                        </div>
                      )}

                      {/* Content Bubble */}
                      <div className={`rounded-2xl p-4.5 ${
                        isSystem 
                          ? 'bg-slate-900/60 border border-white/5 text-slate-400 text-xs py-3 px-6' 
                          : isAgent 
                            ? 'bg-slate-900/90 border border-white/5 text-slate-200 leading-relaxed font-sans' 
                            : 'bg-indigo-600/10 border border-indigo-500/20 text-indigo-200'
                      }`}>
                        {/* Format Message override markers beautifully */}
                        {msg.content.startsWith('[HUMAN SUPERVISOR OVERRIDE]') ? (
                          <div className="space-y-1.5">
                            <span className="text-[10px] font-bold tracking-wider font-mono text-amber-400 uppercase bg-amber-400/10 px-2 py-0.5 rounded border border-amber-400/20">
                              Human Override Applied
                            </span>
                            <p className="font-mono text-sm italic text-amber-300">
                              {msg.content.replace('[HUMAN SUPERVISOR OVERRIDE]: ', '')}
                            </p>
                          </div>
                        ) : msg.content.startsWith('[SYSTEM]:') ? (
                          <div className="space-y-1">
                            <div className="flex items-center gap-1.5 text-red-400 font-bold font-mono text-xs">
                              <AlertTriangle className="w-3.5 h-3.5" />
                              SYSTEM ESCALATION
                            </div>
                            <p className="text-slate-300 text-sm">{msg.content}</p>
                          </div>
                        ) : (
                          <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                        )}

                        {/* Citation badges */}
                        {isAgent && index > 0 && (
                          <div className="mt-3 pt-3 border-t border-white/5 flex flex-wrap gap-2">
                            <span className="text-[10px] uppercase font-mono tracking-wider text-slate-500 flex items-center gap-1">
                              <Layers className="w-3 h-3" />
                              Grounded sources:
                            </span>
                            {/* Simple extraction of sources from text for badges */}
                            {msg.content.includes('[Source:') ? (
                              Array.from(msg.content.matchAll(/\[Source:\s*([^\]]+)\]/g)).map((match, idx) => (
                                <span key={idx} className="text-[10px] font-mono font-medium px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-500/25">
                                  {match[1]}
                                </span>
                              ))
                            ) : (
                              <span className="text-[10px] font-mono text-slate-500 italic">No direct context vector used</span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}

                {/* Agent is thinking step indicator */}
                {isChatLoading && (
                  <div className="flex gap-3 max-w-[85%] mr-auto items-start animate-pulse">
                    <div className="p-2 rounded-xl bg-slate-900 border border-cyan-500/20 text-cyan-400">
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    </div>
                    <div className="bg-slate-900/90 border border-white/5 rounded-2xl p-4 space-y-3 w-72">
                      <div className="h-4 bg-slate-800 rounded w-3/4"></div>
                      <div className="h-3 bg-slate-800 rounded w-5/6"></div>
                    </div>
                  </div>
                )}

                {/* Show LangGraph execution trace */}
                {!isChatLoading && chatSteps.length > 0 && (
                  <div className="mx-auto w-full max-w-2xl bg-slate-950/60 border border-white/5 rounded-2xl p-4.5 mt-4 space-y-3">
                    <div className="flex items-center gap-2 border-b border-white/5 pb-2 text-xs font-semibold text-slate-400 uppercase tracking-widest font-mono">
                      <Terminal className="w-4 h-4 text-cyan-400" />
                      LangGraph Step-by-Step Tracing
                    </div>
                    
                    <div className="space-y-3 max-h-48 overflow-y-auto pr-2">
                      {chatSteps.map((step, idx) => (
                        <div key={idx} className="text-xs font-mono border-l-2 border-cyan-500/30 pl-3 py-0.5 space-y-1">
                          <div className="flex items-center justify-between text-slate-400">
                            <span className="text-cyan-400 font-bold">Node: {step.node}</span>
                            <span className="text-[10px] text-slate-600">Step {idx + 1}</span>
                          </div>
                          <p className="text-slate-300 text-[11px] leading-relaxed">{step.message}</p>
                          
                          {/* If step has tool calls */}
                          {step.tool_calls && step.tool_calls.map((tc, tcIdx) => (
                            <div key={tcIdx} className="bg-slate-900 border border-white/5 rounded p-2 mt-1 space-y-1 text-[10px]">
                              <div className="text-indigo-400 font-semibold">Tool Call: {tc.name}</div>
                              <pre className="text-slate-400 overflow-x-auto whitespace-pre-wrap">{JSON.stringify(tc.args, null, 2)}</pre>
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div ref={chatEndRef} />
              </div>

              {/* Error Box */}
              {chatError && (
                <div className="mx-4 mb-2 p-3 rounded-xl border border-red-500/20 bg-red-500/10 text-red-300 text-xs flex items-center gap-2">
                  <AlertCircle className="w-4.5 h-4.5 shrink-0" />
                  <span>{chatError}</span>
                </div>
              )}

              {/* Human Escalation Controller */}
              {needsHuman && (
                <div className="mx-4 mb-4 p-4.5 rounded-2xl border border-amber-500/30 bg-amber-500/5 glow-secondary space-y-3.5">
                  <div className="flex items-start gap-3">
                    <div className="p-2 rounded-xl bg-amber-500/15 border border-amber-500/30 text-amber-400 shrink-0">
                      <AlertTriangle className="w-5 h-5 animate-pulse" />
                    </div>
                    <div>
                      <h3 className="text-sm font-semibold text-amber-300 font-outfit">Escalation Node Triggered</h3>
                      <p className="text-xs text-slate-400 mt-0.5 leading-relaxed">
                        The LLM identified an request requiring manual oversight (contains request for "human" or "escalate"). LangGraph execution is currently paused. Please review or override below.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <input 
                      type="text" 
                      value={humanGuidance}
                      onChange={(e) => setHumanGuidance(e.target.value)}
                      placeholder="Type supervisor guidance or answers (e.g. 'Add document citation for new agent configs')"
                      className="flex-1 bg-slate-900/80 border border-amber-500/20 rounded-xl px-4 text-xs focus:outline-none focus:border-amber-400 text-amber-100 placeholder-amber-600/50 focus:ring-1 focus:ring-amber-400"
                      onKeyDown={(e) => e.key === 'Enter' && handleResolveEscalation()}
                    />
                    <button
                      onClick={handleResolveEscalation}
                      className="px-4.5 py-2.5 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-semibold text-xs transition duration-200 flex items-center gap-1.5 shadow-md shadow-amber-500/10 hover:shadow-amber-500/25 shrink-0"
                    >
                      Override
                      <CornerDownLeft className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              )}

              {/* Sample prompts / Action chips */}
              {messages.length === 1 && !isChatLoading && (
                <div className="p-4 border-t border-white/5 bg-slate-950/40">
                  <span className="text-[10px] font-bold font-mono text-slate-500 uppercase tracking-widest block mb-2 px-1">Suggested Inquiries</span>
                  <div className="flex flex-wrap gap-2">
                    {[
                      { text: "Why do AI systems use cross-encoder reranking?", icon: <Sparkles className="w-3 h-3 text-cyan-400" /> },
                      { text: "Explain Reciprocal Rank Fusion", icon: <Layers className="w-3 h-3 text-indigo-400" /> },
                      { text: "I need human help to debug qdrant", icon: <AlertTriangle className="w-3 h-3 text-amber-400" /> }
                    ].map((chip, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleSendMessage(chip.text)}
                        className="flex items-center gap-1.5 px-3.5 py-2 text-xs rounded-xl border border-white/5 hover:border-cyan-500/20 bg-slate-900/60 hover:bg-slate-900 text-slate-300 hover:text-cyan-300 transition duration-200"
                      >
                        {chip.icon}
                        {chip.text}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Chat Input */}
              <div className="p-4 border-t border-white/5 bg-slate-950/80">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    placeholder={!isConnected ? "Wait for backend core initialization..." : "Query the Aethelgard knowledge base..."}
                    className="flex-1 bg-slate-900 border border-white/5 focus:border-cyan-500/50 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-1 focus:ring-cyan-500/50 transition duration-200 placeholder-slate-500 text-white disabled:opacity-50"
                    onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                    disabled={isChatLoading || needsHuman || !isConnected}
                  />
                  <button
                    onClick={() => handleSendMessage()}
                    className="p-3 bg-gradient-to-tr from-cyan-500 to-indigo-600 hover:brightness-110 text-slate-950 font-bold rounded-xl transition duration-200 shadow-md shadow-cyan-500/10 flex items-center justify-center shrink-0 disabled:opacity-50 disabled:brightness-100"
                    disabled={isChatLoading || needsHuman || !inputMessage.trim() || !isConnected}
                  >
                    <Send className="w-5 h-5 text-slate-900" />
                  </button>
                </div>
              </div>

            </div>
          )}

          {/* TAB 2: Document Ingestion */}
          {activeTab === 'ingest' && (
            <div className="space-y-6">
              
              {/* Form Card */}
              <div className="glass-card rounded-3xl p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Database className="w-5 h-5 text-cyan-400" />
                  <h2 className="text-lg font-bold font-outfit text-white">Embed and Ingest Document</h2>
                </div>

                <form onSubmit={handleIngest} className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <label className="text-xs font-semibold text-slate-400 uppercase tracking-widest font-mono">Source Identifier</label>
                      <input 
                        type="text" 
                        value={ingestSource}
                        onChange={(e) => setIngestSource(e.target.value)}
                        placeholder="e.g. LLM_Hallucinations_Paper.pdf"
                        className="w-full bg-slate-900 border border-white/5 focus:border-cyan-500/50 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-1 focus:ring-cyan-500/50 transition duration-200 text-white placeholder-slate-600"
                        required
                      />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-xs font-semibold text-slate-400 uppercase tracking-widest font-mono">Tags / Metadata (comma separated)</label>
                      <input 
                        type="text" 
                        value={ingestTags}
                        onChange={(e) => setIngestTags(e.target.value)}
                        placeholder="e.g. artificial_intelligence, rerank, math"
                        className="w-full bg-slate-900 border border-white/5 focus:border-cyan-500/50 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-1 focus:ring-cyan-500/50 transition duration-200 text-white placeholder-slate-600"
                      />
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-xs font-semibold text-slate-400 uppercase tracking-widest font-mono">Research Text / Knowledge Content</label>
                    <textarea 
                      rows={5}
                      value={ingestText}
                      onChange={(e) => setIngestText(e.target.value)}
                      placeholder="Paste research paper paragraphs, documentation chapters, or raw engineering specifications..."
                      className="w-full bg-slate-900 border border-white/5 focus:border-cyan-500/50 rounded-xl p-4 text-sm focus:outline-none focus:ring-1 focus:ring-cyan-500/50 transition duration-200 text-white placeholder-slate-600 resize-none"
                      required
                    />
                  </div>

                  {ingestResult && (
                    <div className={`p-3.5 rounded-xl text-xs flex items-center gap-2 border ${
                      ingestResult.type === 'success' 
                        ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300' 
                        : 'border-red-500/20 bg-red-500/10 text-red-300'
                    }`}>
                      {ingestResult.type === 'success' ? <CheckCircle className="w-4.5 h-4.5 shrink-0" /> : <AlertCircle className="w-4.5 h-4.5 shrink-0" />}
                      <span>{ingestResult.msg}</span>
                    </div>
                  )}

                  <button
                    type="submit"
                    className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-indigo-600 hover:brightness-110 text-slate-950 font-bold rounded-xl text-sm transition duration-200 flex items-center justify-center gap-2 shadow-md shadow-cyan-500/10 disabled:opacity-50"
                    disabled={isIngesting || !isConnected}
                  >
                    {isIngesting ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin text-slate-900" />
                        Generating dense/sparse embeddings...
                      </>
                    ) : (
                      <>
                        <Plus className="w-4 h-4 text-slate-900" />
                        Ingest Knowledge
                      </>
                    )}
                  </button>
                </form>
              </div>

              {/* List Card */}
              <div className="glass-card rounded-3xl p-6 space-y-4">
                <div className="flex items-center justify-between border-b border-white/5 pb-3">
                  <div className="flex items-center gap-2">
                    <FileText className="w-5 h-5 text-indigo-400" />
                    <h2 className="text-lg font-bold font-outfit text-white">Ingested Document Points ({ingestedDocs.length})</h2>
                  </div>
                  <button 
                    onClick={fetchIngestedDocs}
                    className="p-1.5 rounded-lg border border-white/10 hover:bg-white/5 transition text-slate-400 hover:text-white"
                    title="Refresh document list"
                    disabled={isLoadingDocs}
                  >
                    <RefreshCw className={`w-4 h-4 ${isLoadingDocs ? 'animate-spin' : ''}`} />
                  </button>
                </div>

                {isLoadingDocs ? (
                  <div className="text-center py-8 text-xs font-mono text-slate-500 animate-pulse">
                    Retrieving database points from Qdrant local cluster...
                  </div>
                ) : ingestedDocs.length === 0 ? (
                  <div className="text-center py-8 text-xs font-mono text-slate-500 border border-dashed border-white/5 rounded-2xl">
                    No documents found. Ingest a document above or sync backend.
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-80 overflow-y-auto pr-2">
                    {ingestedDocs.map((doc) => (
                      <div key={doc.id} className="bg-slate-950/60 border border-white/5 hover:border-cyan-500/10 rounded-xl p-4 flex flex-col justify-between gap-3 text-xs group transition duration-200">
                        <div className="space-y-2">
                          <div className="flex items-center justify-between font-mono text-[10px] text-slate-500">
                            <span className="text-cyan-400/80 font-semibold truncate max-w-[70%]">{doc.source}</span>
                            <span className="truncate w-20 text-right">{doc.id.slice(0, 8)}...</span>
                          </div>
                          <p className="text-slate-300 line-clamp-3 leading-relaxed text-[11px] font-sans">{doc.text}</p>
                        </div>

                        {/* Tags representation */}
                        {doc.metadata?.tags && (
                          <div className="flex flex-wrap gap-1.5 mt-1">
                            {doc.metadata.tags.map((tag: string, idx: number) => (
                              <span key={idx} className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-900 border border-white/5 text-slate-400">
                                {tag}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

            </div>
          )}

          {/* TAB 3: Direct Search Sandbox */}
          {activeTab === 'search' && (
            <div className="space-y-6">
              
              {/* Search Control Card */}
              <div className="glass-card rounded-3xl p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Search className="w-5 h-5 text-cyan-400" />
                  <h2 className="text-lg font-bold font-outfit text-white">Hybrid Vector Search Sandbox</h2>
                </div>

                <form onSubmit={handleSearch} className="space-y-4">
                  <div className="flex gap-2">
                    <input 
                      type="text" 
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Type a research query to retrieve vector points..."
                      className="flex-1 bg-slate-900 border border-white/5 focus:border-cyan-500/50 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-1 focus:ring-cyan-500/50 transition duration-200 text-white placeholder-slate-500"
                      required
                    />
                    
                    <select
                      value={searchLimit}
                      onChange={(e) => setSearchLimit(Number(e.target.value))}
                      className="bg-slate-900 border border-white/5 rounded-xl px-3 py-3 text-xs text-slate-300 focus:outline-none focus:border-cyan-500"
                    >
                      <option value={3}>Top 3</option>
                      <option value={5}>Top 5</option>
                      <option value={10}>Top 10</option>
                    </select>

                    <button
                      type="submit"
                      className="px-5 py-3 bg-gradient-to-r from-cyan-500 to-indigo-600 hover:brightness-110 text-slate-950 font-bold rounded-xl text-sm transition duration-200 flex items-center justify-center shrink-0 shadow-md shadow-cyan-500/10"
                      disabled={isSearching || !isConnected}
                    >
                      {isSearching ? <RefreshCw className="w-4 h-4 animate-spin text-slate-900" /> : <Search className="w-4 h-4 text-slate-900" />}
                    </button>
                  </div>
                </form>
              </div>

              {/* Search Results Display */}
              <div className="glass-card rounded-3xl p-6 space-y-4">
                <div className="flex items-center gap-2 border-b border-white/5 pb-3">
                  <Layers className="w-5 h-5 text-indigo-400" />
                  <h2 className="text-lg font-bold font-outfit text-white">Retrieval & Reranking Results</h2>
                </div>

                {searchError && (
                  <div className="p-3.5 rounded-xl border border-red-500/20 bg-red-500/10 text-red-300 text-xs flex items-center gap-2">
                    <AlertCircle className="w-4.5 h-4.5 shrink-0" />
                    <span>{searchError}</span>
                  </div>
                )}

                {isSearching ? (
                  <div className="text-center py-12 space-y-3">
                    <RefreshCw className="w-8 h-8 animate-spin text-cyan-400 mx-auto" />
                    <div className="text-xs font-mono text-slate-400">
                      1. Generating dense (BGE) and sparse (SPLADE) vector queries...<br />
                      2. Fetching from Qdrant and fusing with Reciprocal Rank Fusion (RRF)...<br />
                      3. Score evaluation with FlashRank Cross-Encoder reranker...
                    </div>
                  </div>
                ) : searchResults.length === 0 ? (
                  <div className="text-center py-12 text-slate-500 text-sm">
                    {searchQuery ? 'No documents matched this query.' : 'Run a query above to analyze the hybrid-search and reranker score distribution.'}
                  </div>
                ) : (
                  <div className="space-y-4">
                    {searchResults.map((result, idx) => (
                      <div key={idx} className="bg-slate-950/60 border border-white/5 rounded-2xl p-5 hover:border-cyan-500/10 transition duration-200 flex flex-col md:flex-row gap-4 justify-between items-start md:items-center">
                        <div className="space-y-2 flex-1">
                          <div className="flex flex-wrap items-center gap-2 font-mono text-[10px]">
                            <span className="font-bold text-cyan-400 uppercase bg-cyan-400/5 border border-cyan-400/20 px-2 py-0.5 rounded">
                              Source: {result.metadata.source || 'Unknown'}
                            </span>
                            <span className="text-slate-500">
                              Relevance Index: {idx + 1}
                            </span>
                          </div>
                          
                          <p className="text-sm text-slate-200 leading-relaxed font-sans">{result.text}</p>
                          
                          {result.metadata.timestamp && (
                            <span className="text-[10px] text-slate-500 font-mono block">
                              Ingested: {new Date(result.metadata.timestamp).toLocaleString()}
                            </span>
                          )}
                        </div>

                        {/* Reranker Score Circular Indicator */}
                        <div className="flex flex-col items-center justify-center bg-slate-900 border border-white/10 rounded-2xl p-4.5 min-w-[110px] shrink-0 text-center">
                          <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold font-mono">Ranker Score</span>
                          <span className="text-lg font-extrabold text-cyan-400 font-outfit mt-1">
                            {result.score.toFixed(4)}
                          </span>
                          <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden mt-2 border border-white/5">
                            <div 
                              className="bg-gradient-to-r from-cyan-400 to-indigo-500 h-full"
                              style={{ width: `${Math.min(100, result.score * 100)}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

            </div>
          )}

        </main>
      </div>

      {/* Footer */}
      <footer className="border-t border-white/5 bg-slate-950/40 py-6 text-center text-xs text-slate-500 font-mono">
        <p>Aethelgard Engine Core v1.0.0 | Pair-programmed with Antigravity AI</p>
      </footer>
    </div>
  )
}

export default App
