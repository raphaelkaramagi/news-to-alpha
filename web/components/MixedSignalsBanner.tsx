export function MixedSignalsBanner() {
  return (
    <div
      className="rounded-lg border border-warn/40 bg-warn/10 px-4 py-2 text-center text-sm font-medium text-warn"
      role="status"
    >
      Mixed signals — component models disagree on direction for this date.
    </div>
  );
}
