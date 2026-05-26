import { Metadata } from "next";
import { generateMetadataTemplate } from "@/lib/metadata";
import BusinessDashboard from "@/components/business/business-dashboard";

export async function generateMetadata(): Promise<Metadata> {
  return generateMetadataTemplate({
    title: "Business Dashboard | Cost Savvy Health",
    description:
      "Private dashboard for employers, providers, and payers to manage data uploads and organization updates.",
    keywords: [
      "business dashboard",
      "provider dashboard",
      "payer dashboard",
      "hospital data upload",
      "CPT data management",
    ],
    url: "https://www.costsavvy.health/dashboard/business",
  });
}

export default function Page() {
  return <BusinessDashboard />;
}
