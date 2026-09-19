import type { NextConfig } from "next";

// FastAPI (rev.serve) is proxied under /rev (and /kev) so the browser never deals with CORS or ports.
const REV_API = process.env.REV_API ?? process.env.KEV_API ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  reactCompiler: true,
  devIndicators: false,
  allowedDevOrigins: ["127.0.0.1"],
  async rewrites() {
    return [
      { source: "/rev/:path*", destination: `${REV_API}/:path*` },
      { source: "/kev/:path*", destination: `${REV_API}/:path*` },
    ];
  },
};

export default nextConfig;
