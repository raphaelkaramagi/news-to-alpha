import type { MetadataRoute } from "next";

/** PWA manifest — icons in public/ are generated from web/app/icon.png source. */
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Stock Price and Sentiment Predictor",
    short_name: "Stock Predictor",
    description:
      "Next-session stock direction forecasts from price + news sentiment.",
    start_url: "/",
    display: "standalone",
    background_color: "#09090b",
    theme_color: "#09090b",
    icons: [
      { src: "/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png" },
    ],
  };
}
