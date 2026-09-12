"use client";

export function ThemeToggle() {
  function toggleTheme() {
    const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("cache-flow-theme", theme); } catch { /* Theme still works when storage is blocked. */ }
  }

  return <button type="button" className="theme-toggle" onClick={toggleTheme}>
    <span className="theme-light"><span aria-hidden="true">◐</span> Light<span className="sr-only"> — switch to dark theme</span></span>
    <span className="theme-dark"><span aria-hidden="true">◑</span> Dark<span className="sr-only"> — switch to light theme</span></span>
  </button>;
}
