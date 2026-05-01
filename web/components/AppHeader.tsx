import Link from "next/link";

const nav = [
  { href: "/", label: "Markets" },
  { href: "/pipeline", label: "Pipeline" },
];

export function AppHeader() {
  return (
    <header className="border-b border-border bg-surface/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-6 px-4 py-4 sm:px-6">
        <Link href="/" className="font-sans text-lg font-semibold tracking-tight">
          News-to-<span className="text-accent">Alpha</span>
        </Link>
        <nav className="flex flex-wrap items-center gap-1 sm:gap-4" aria-label="Primary">
          {nav.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="rounded-lg px-2.5 py-1.5 text-sm font-medium text-muted transition hover:bg-surface-2 hover:text-foreground sm:px-3"
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
