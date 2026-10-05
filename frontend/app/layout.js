import "./globals.css";
import { LanguageProvider } from "./i18n";

export const metadata = {
  title: "Ontology Builder — LogosAI",
  description: "Upload data and let an LLM build an ontology — graph, map, semantic search, training-data export.",
};

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        <LanguageProvider>{children}</LanguageProvider>
      </body>
    </html>
  );
}
