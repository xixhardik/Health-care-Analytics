/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Slice images come from the FastAPI backend as plain PNG responses and are
  // rendered with a normal <img>, so next/image optimisation is not involved.
  poweredByHeader: false,
  eslint: { ignoreDuringBuilds: true },
};

export default nextConfig;
