import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-lg px-4 py-24 text-center">
      <h1 className="text-2xl font-bold">Not found</h1>
      <p className="mt-2 text-muted">
        That ticker is not in the tracked universe.
      </p>
      <Link href="/" className="mt-6 inline-block text-accent hover:underline">
        Back to Markets
      </Link>
    </div>
  );
}
