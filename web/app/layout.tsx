import type { Metadata } from "next";
import { Outfit, JetBrains_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";
import { AppHeader } from "@/components/AppHeader";

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-outfit",
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
});

export const metadata: Metadata = {
  title: "News-to-Alpha",
  description:
    "Next-session market direction from news and price signals — lightweight Next.js UI.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${outfit.variable} ${jetbrains.variable} min-h-screen bg-background font-sans antialiased`}
      >
        <AppHeader />
        {children}
        <Link
          href="/pipeline"
          className="fixed bottom-5 right-5 flex h-11 w-11 items-center justify-center rounded-full border border-border bg-surface text-muted shadow-lg transition hover:border-accent hover:text-foreground"
          aria-label="Pipeline and jobs"
          title="Pipeline"
        >
          <span className="text-lg" aria-hidden>
            ⚙
          </span>
        </Link>
      </body>
    </html>
  );
}
