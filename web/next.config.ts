import path from "path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Monorepo: avoid Turbopack picking up a stray lockfile in the parent folder
  turbopack: {
    root: path.join(__dirname),
  },
  async redirects() {
    return [
      {
        source: "/ticker/:symbol",
        destination: "/t/:symbol",
        permanent: true,
      },
      {
        source: "/pipeline",
        destination: "/status",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
