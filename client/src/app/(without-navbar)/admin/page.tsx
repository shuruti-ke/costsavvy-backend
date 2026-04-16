import { Metadata } from "next";
import { generateMetadataTemplate } from "@/lib/metadata";
import AdminConsole from "@/components/admin/admin-console";

export async function generateMetadata(): Promise<Metadata> {
  return generateMetadataTemplate({
    title: "Platform Admin | Cost Savvy Health",
    description:
      "Administrator console for managing users, editorial content, pricing data, and correspondence across Cost Savvy Health.",
    keywords: [
      "admin",
      "platform admin",
      "healthcare operations",
      "content management",
      "user management",
      "healthcare data",
    ],
    url: "https://www.costsavvy.health/admin",
  });
}

export default function AdminPage() {
  return <AdminConsole />;
}
