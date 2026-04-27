import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Online Buying Assistant",
  description: "Production RAG console for product ingestion, evidence-backed answers, and eval runs.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
