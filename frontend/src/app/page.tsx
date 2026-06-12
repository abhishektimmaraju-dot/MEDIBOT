"use client";

import React, { useState, useEffect, useRef } from "react";

const BACKEND_URL = "http://127.0.0.1:8000";

const DEMO_ACCOUNTS = [
  { username: "dr.mehta", label: "Doctor (Dr. Mehta)", role: "doctor", bg: "bg-blue-50 border-blue-200 text-blue-800" },
  { username: "nurse.priya", label: "Nurse (Nurse Priya)", role: "nurse", bg: "bg-emerald-50 border-emerald-200 text-emerald-800" },
  { username: "billing.ravi", label: "Billing Executive (Billing Ravi)", role: "billing_executive", bg: "bg-amber-50 border-amber-200 text-amber-800" },
  { username: "tech.anand", label: "Technician (Tech Anand)", role: "technician", bg: "bg-purple-50 border-purple-200 text-purple-800" },
  { username: "admin.sys", label: "Admin (Admin Sys)", role: "admin", bg: "bg-rose-50 border-rose-200 text-rose-800" }
];

const COLLECTION_LABELS: Record<string, string> = {
  general: "General (HR, Leave Policy, Code of Conduct)",
  clinical: "Clinical (Drug Formulary, Treatment Protocols)",
  nursing: "Nursing (ICU Procedures, Infection Control)",
  billing: "Billing & Insurance (Billing Codes, claims)",
  equipment: "Medical Equipment (Manuals, Calibration)"
};

interface Message {
  sender: "user" | "bot";
  text: string;
  retrievalType?: string;
  sources?: Array<{ source_document: string; section_title: string; collection: string }>;
}


export default function Home() {
  const [token, setToken] = useState<string | null>(null);
  const [role, setRole] = useState<string | null>(null);
  const [userName, setUserName] = useState<string | null>(null);
  const [allowedCollections, setAllowedCollections] = useState<string[]>([]);
  
  const [selectedUser, setSelectedUser] = useState<string>("dr.mehta");
  const [password, setPassword] = useState<string>("password");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [inputVal, setInputVal] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Scroll to bottom whenever messages list updates
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);
    try {
      const res = await fetch(`${BACKEND_URL}/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: selectedUser, password })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Login failed");
      }
      const data = await res.json();
      setToken(data.access_token);
      setRole(data.role);
      setUserName(data.name);
      
      // Load allowed collections
      let collections: string[] = [];
      const colRes = await fetch(`${BACKEND_URL}/collections/${data.role}`);
      if (colRes.ok) {
        const colData = await colRes.json();
        collections = colData.collections;
        setAllowedCollections(collections);
      }
      
      // Welcome message
      setMessages([
        {
          sender: "bot",
          text: `Welcome, ${data.name}! I am MediBot. You are logged in with the role of **${data.role.replace("_", " ").toUpperCase()}**. You have access to the **${data.role === "admin" ? "All" : collections.join(", ")}** collections. How can I help you today?`
        }
      ]);
    } catch (err: any) {

      setErrorMsg(err.message);
    }
  };

  const handleLogout = () => {
    setToken(null);
    setRole(null);
    setUserName(null);
    setAllowedCollections([]);
    setMessages([]);
    setInputVal("");
  };

  const sendMessage = async (text: string) => {
    if (!text.trim() || loading) return;
    
    // Add user message
    const userMsg: Message = { sender: "user", text };
    setMessages(prev => [...prev, userMsg]);
    setInputVal("");
    setLoading(true);

    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ question: text })
      });
      if (!res.ok) {
        throw new Error("Chat request failed");
      }
      const data = await res.json();
      setMessages(prev => [...prev, {
        sender: "bot",
        text: data.answer,
        retrievalType: data.retrieval_type,
        sources: data.sources
      }]);
    } catch (err: any) {
      setMessages(prev => [...prev, {
        sender: "bot",
        text: "Error contacting the server. Please check that the backend API is running."
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleSamplePrompt = (promptText: string) => {
    sendMessage(promptText);
  };

  // Render Login Page
  if (!token) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-900 px-4 py-12 text-slate-100">
        <div className="w-full max-w-lg space-y-8 rounded-2xl border border-slate-800 bg-slate-950 p-8 shadow-2xl">
          <div className="text-center">
            <h2 className="text-4xl font-extrabold tracking-tight text-white bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
              MediBot Login
            </h2>
            <p className="mt-2 text-sm text-slate-400">
              Select one of the demo users below to simulate access controls.
            </p>
          </div>
          
          <form onSubmit={handleLogin} className="mt-8 space-y-6">
            <div className="space-y-4">
              <label className="block text-sm font-medium text-slate-300">Demo Account Profile</label>
              <div className="grid grid-cols-1 gap-2">
                {DEMO_ACCOUNTS.map((acc) => (
                  <button
                    key={acc.username}
                    type="button"
                    onClick={() => setSelectedUser(acc.username)}
                    className={`flex items-center justify-between rounded-lg border p-4 text-left transition-all ${
                      selectedUser === acc.username
                        ? "border-blue-500 bg-slate-900 ring-2 ring-blue-500/20"
                        : "border-slate-800 bg-slate-950 hover:bg-slate-900"
                    }`}
                  >
                    <div>
                      <p className="font-semibold text-white">{acc.label}</p>
                      <p className="text-xs text-slate-500">Username: {acc.username}</p>
                    </div>
                    <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wider ${acc.bg}`}>
                      {acc.role.replace("_", " ")}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium text-slate-300">Password</label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-lg border border-slate-800 bg-slate-950 px-4 py-3 text-white focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
              />
            </div>

            {errorMsg && (
              <div className="rounded-lg border border-red-900 bg-red-950/50 p-4 text-sm text-red-400">
                {errorMsg}
              </div>
            )}

            <button
              type="submit"
              className="w-full rounded-lg bg-gradient-to-r from-blue-500 to-emerald-500 py-3 text-center font-bold text-white shadow-lg shadow-blue-500/20 hover:from-blue-600 hover:to-emerald-600 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
            >
              Sign In to MediBot
            </button>
          </form>
        </div>
      </div>
    );
  }

  // Allowed vs restricted list for sidebar
  const allCollectionsList = ["general", "clinical", "nursing", "billing", "equipment"];

  const getSamplePromptsForRole = () => {
    switch (role) {
      case "doctor":
        return [
          "What is the standard treatment protocol for NSTEMI?",
          "Are cashless claims processed at emergency admissions?",
          "Show me HDFC Ergo cashless pre-authorisation timelines from the billing guides."
        ];
      case "nurse":
        return [
          "What is the emergency pre-authorisation request timeline?",
          "Explain the ICU standard protocols for infection control.",
          "Show me Star Health cashless pre-authorisation timelines from the billing codes."
        ];
      case "billing_executive":
        return [
          "How many cashless claims are pending in cardiology department?",
          "What is the total claimed amount across all cashless claims?",
          "What are the steps for battery replacement on an infusion pump?"
        ];
      case "technician":
        return [
          "What are the calibration steps for the infusion pump?",
          "Explain the preventive maintenance schedules.",
          "How many claims were approved last month?"
        ];
      case "admin":
      default:
        return [
          "How many claims were escalated last month?",
          "What is the standard treatment protocol for NSTEMI?",
          "What is the standard cashless pre-auth SLA for Bajaj Allianz?",
          "Which equipment category has the most open maintenance tickets?"
        ];
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100">
      {/* Sidebar */}
      <aside className="flex w-80 flex-col border-r border-slate-800 bg-slate-950 p-6">
        <div className="flex items-center gap-3 border-b border-slate-800 pb-6">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-blue-500 to-emerald-500 flex items-center justify-center font-bold text-white text-lg">
            M
          </div>
          <div>
            <h1 className="font-bold text-lg text-white tracking-tight">MediBot RAG</h1>
            <p className="text-xs text-slate-500">MediAssist Operations</p>
          </div>
        </div>

        {/* User profile */}
        <div className="mt-6 rounded-xl border border-slate-800 bg-slate-900/50 p-4">
          <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider">Active Profile</p>
          <p className="font-bold text-white mt-1">{userName}</p>
          <span className="inline-block mt-2 rounded-full bg-blue-900/40 border border-blue-800 text-blue-300 px-3 py-0.5 text-xs font-semibold uppercase tracking-wider">
            {role?.replace("_", " ")}
          </span>
        </div>

        {/* Allowed Collections section */}
        <div className="mt-8 flex-1 space-y-4 overflow-y-auto">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Document Access Matrix</h3>
          <ul className="space-y-2">
            {allCollectionsList.map((col) => {
              const isAllowed = allowedCollections.includes(col) || role === "admin";
              return (
                <li
                  key={col}
                  className={`flex items-center justify-between rounded-lg border px-4 py-3 text-sm transition-all ${
                    isAllowed
                      ? "border-emerald-950 bg-emerald-950/20 text-slate-100"
                      : "border-slate-900 bg-slate-950/40 text-slate-600"
                  }`}
                >
                  <span className="font-medium">{COLLECTION_LABELS[col] || col}</span>
                  {isAllowed ? (
                    <span className="text-xs text-emerald-400 font-semibold bg-emerald-900/20 px-2 py-0.5 rounded-full border border-emerald-900/40">Open</span>
                  ) : (
                    <span className="text-xs text-slate-600 font-semibold bg-slate-900/10 px-2 py-0.5 rounded-full border border-slate-900/30">Locked</span>
                  )}
                </li>
              );
            })}
          </ul>
        </div>

        {/* Logout */}
        <button
          onClick={handleLogout}
          className="mt-6 w-full rounded-lg border border-slate-800 bg-slate-950 py-3 text-center text-sm font-semibold text-slate-400 hover:bg-slate-900 hover:text-white transition-all"
        >
          Logout
        </button>
      </aside>

      {/* Main Chat Panel */}
      <main className="flex flex-1 flex-col bg-slate-950">
        {/* Chat Header */}
        <header className="flex h-16 items-center justify-between border-b border-slate-800 bg-slate-950 px-8 shadow-sm">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <p className="text-sm font-medium text-slate-300">Connected to local Qdrant & SQLite DB</p>
          </div>
        </header>

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          {messages.map((msg, index) => {
            const isBot = msg.sender === "bot";
            const isRejection = isBot && msg.text.includes("Access Denied");
            
            return (
              <div
                key={index}
                className={`flex ${isBot ? "justify-start" : "justify-end"}`}
              >
                <div
                  className={`max-w-2xl rounded-2xl px-5 py-4 shadow-md leading-relaxed ${
                    !isBot
                      ? "bg-blue-600 text-white rounded-br-none"
                      : isRejection
                      ? "border border-red-950 bg-red-950/30 text-red-300 rounded-bl-none"
                      : "border border-slate-800 bg-slate-900/60 text-slate-100 rounded-bl-none"
                  }`}
                >
                  {/* Message Header */}
                  <div className="flex items-center justify-between gap-6 border-b border-slate-800 pb-2 mb-2 text-xs text-slate-500">
                    <span className="font-semibold uppercase tracking-wider text-slate-400">
                      {isBot ? "MediBot" : "You"}
                    </span>
                    {isBot && msg.retrievalType && (
                      <span className={`rounded px-1.5 py-0.5 font-bold uppercase tracking-widest ${
                        msg.retrievalType === "sql_rag" ? "bg-purple-900/40 text-purple-300 border border-purple-800" : "bg-blue-900/40 text-blue-300 border border-blue-800"
                      }`}>
                        {msg.retrievalType === "sql_rag" ? "SQL RAG" : "Hybrid RAG"}
                      </span>
                    )}
                  </div>

                  {/* Message Text */}
                  <p className="whitespace-pre-wrap text-sm">{msg.text}</p>

                  {/* Sources display */}
                  {isBot && msg.sources && msg.sources.length > 0 && (
                    <div className="mt-4 pt-3 border-t border-slate-800 text-xs text-slate-400">
                      <p className="font-semibold text-slate-300 mb-1">Sources consulted:</p>
                      <ul className="list-disc pl-4 space-y-1">
                        {msg.sources.map((s, sIdx) => (
                          <li key={sIdx}>
                            <span className="font-medium text-slate-200">{s.source_document}</span> (Section: {s.section_title}) 
                            <span className="ml-2 inline-block rounded bg-slate-800 px-1 py-0.5 text-[10px] text-slate-400 uppercase">{s.collection}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
          
          {loading && (
            <div className="flex justify-start">
              <div className="flex items-center gap-2 max-w-xs rounded-2xl border border-slate-800 bg-slate-900/60 px-5 py-4 text-slate-400 text-sm rounded-bl-none shadow-md">
                <svg className="animate-spin h-5 w-5 text-blue-500" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                MediBot is processing query...
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Sample prompts */}
        <div className="px-8 py-3 bg-slate-950 border-t border-slate-900 flex flex-wrap gap-2 items-center">
          <span className="text-xs text-slate-500 font-semibold mr-1 uppercase">Sample Prompts:</span>
          {getSamplePromptsForRole().map((prompt, idx) => (
            <button
              key={idx}
              onClick={() => handleSamplePrompt(prompt)}
              className="text-xs border border-slate-800 bg-slate-900/40 text-slate-300 px-3 py-1.5 rounded-full hover:bg-slate-900 hover:border-slate-700 transition-all text-left max-w-sm truncate"
            >
              {prompt}
            </button>
          ))}
        </div>

        {/* Input Bar */}
        <footer className="p-8 bg-slate-950 border-t border-slate-900">
          <div className="relative flex items-center">
            <input
              type="text"
              placeholder="Ask MediBot a question (e.g. clinical guidance, leave policy, billing rules, database stats)..."
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendMessage(inputVal)}
              className="w-full rounded-xl border border-slate-850 bg-slate-900 px-6 py-4 text-sm text-slate-100 placeholder-slate-500 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/10 pr-20"
            />
            <button
              onClick={() => sendMessage(inputVal)}
              disabled={loading || !inputVal.trim()}
              className="absolute right-3 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:bg-slate-800 disabled:text-slate-500 transition-all"
            >
              Send
            </button>
          </div>
        </footer>
      </main>
    </div>
  );
}
