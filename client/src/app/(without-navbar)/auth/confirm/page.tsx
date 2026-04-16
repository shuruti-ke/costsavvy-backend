import { Metadata } from "next";
import { generateMetadataTemplate } from "@/lib/metadata";
import EmailConfirmationView from "@/components/auth/email-confirmation-view";

export async function generateMetadata(): Promise<Metadata> {
  return generateMetadataTemplate({
    title: "Confirm Email | Cost Savvy Health",
    description:
      "Confirm your Cost Savvy Health account and activate access to your dashboard.",
    keywords: [
      "email confirmation",
      "account verification",
      "healthcare platform",
      "dashboard access",
    ],
    url: "https://www.costsavvy.health/auth/confirm",
  });
}

export default async function Page({
  searchParams,
}: {
  searchParams?: Promise<{ token?: string }>;
}) {
  const params = searchParams ? await searchParams : undefined;
  return <EmailConfirmationView token={params?.token || ""} />;
}
