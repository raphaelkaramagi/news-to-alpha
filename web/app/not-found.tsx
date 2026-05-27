import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[40vh] gap-4 text-center">
      <h1 className="text-2xl font-semibold">Not found</h1>
      <p className="text-muted-foreground text-sm">
        That ticker or page doesn&apos;t exist.
      </p>
      <Link
        href="/"
        className="text-sm underline underline-offset-4 text-muted-foreground hover:text-foreground"
      >
        Back to Markets
      </Link>
    </div>
  );
}
