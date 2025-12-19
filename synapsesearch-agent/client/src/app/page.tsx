"use client";
import { useState } from 'react';
import { Search } from 'lucide-react';

export default function Home() {
  const [query, setQuery] = useState("");

  const handleSearch = async () => {
    // Call backend endpoint /api/search
    console.log("Searching for:", query);
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24 bg-slate-950 text-white">
      <h1 className="text-4xl font-bold mb-8">SynapseSearch-Agent 🧠</h1>
      <div className="relative w-full max-w-2xl">
        <input 
          className="w-full p-4 rounded-full bg-slate-900 border border-slate-700 focus:ring-2 focus:ring-blue-500 outline-none"
          placeholder="How do I handle authentication in Next.js?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button onClick={handleSearch} className="absolute right-4 top-4">
          <Search className="text-slate-400 hover:text-white" />
        </button>
      </div>
    </main>
  );
}