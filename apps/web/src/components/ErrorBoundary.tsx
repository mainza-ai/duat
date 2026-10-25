import React from "react";

/** Production safety net: a WebGL/Canvas crash must never blank the whole app. */
export class ViewportErrorBoundary extends React.Component<
  { name: string; children: React.ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error) {
    console.error(`[ViewportErrorBoundary:${this.props.name}]`, error);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex-1 min-h-[480px] flex flex-col items-center justify-center gap-3 bg-[#090A0F] text-slate-300 p-8 text-center">
          <div className="text-sm font-semibold text-white">
            {this.props.name} crashed — the rest of the observatory is unaffected.
          </div>
          <div className="font-mono text-xs text-slate-500 max-w-xl break-words">
            {String(this.state.error?.message ?? this.state.error).slice(0, 300)}
          </div>
          <button
            onClick={() => this.setState({ error: null })}
            className="px-4 py-2 rounded-lg bg-slate-800 border border-slate-700 text-xs font-medium hover:bg-slate-700 transition">
            Reload viewport
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
