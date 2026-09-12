import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CACHE FLOW — A semantic cache laboratory",
  description: "Trace an LLM request. Explore exact and semantic reuse, inspect the cache, and evaluate its quality.",
};
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en" data-theme="light" suppressHydrationWarning><head><script dangerouslySetInnerHTML={{ __html: `try{document.documentElement.dataset.theme=localStorage.getItem("cache-flow-theme")==="dark"?"dark":"light"}catch{document.documentElement.dataset.theme="light"}` }} /></head><body>{children}</body></html>;
}
