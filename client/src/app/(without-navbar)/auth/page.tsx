import { Metadata } from 'next';
import { generateMetadataTemplate } from '@/lib/metadata';

export async function generateMetadata(): Promise<Metadata> {
  return generateMetadataTemplate({
    title: 'Sign In or Sign Up | Cost Savvy Health',
    description: 'Sign in or create a consumer, employer, provider, or payer account for Cost Savvy Health to access the right dashboard and workflow.',
    keywords: [
      'sign in',
      'login',
      'healthcare account',
      'medical cost comparison',
      'healthcare transparency',
      'patient portal',
      'healthcare savings'
    ],
    url: 'https://www.costsavvy.health/auth',
  });
}


import React, { Suspense } from "react";
import { AuthSlider } from "@/components/auth/auth-slider";

export default function Page() {
  return (
    <div className="flex min-h-svh flex-col items-center justify-center gap-6 p-6 md:p-10 bg-[#8C2F5D]">
      <div className="flex w-full max-w-6xl flex-col gap-6">
        <Suspense fallback={<div>Loading...</div>}>
          <AuthSlider />
        </Suspense>
      </div>
    </div>
  );
}
