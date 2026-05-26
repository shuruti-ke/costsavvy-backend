import { Metadata } from 'next';

interface MetadataTemplateProps {
  title?: string;
  description?: string;
  keywords?: string[];
  image?: string;
  url?: string;
}

const DEFAULT_OG_IMAGE = '/icon.png';
const BASE_URL = 'https://www.costsavvy.health';

function normalizeBrand(value: string) {
  return value
    .replace(/Cost Savy Health/g, 'Cost Savvy Health')
    .replace(/cost-savy-health/g, 'cost-savvy-health')
    .replace(/costsavyhealth\.com/gi, 'www.costsavvy.health');
}

export function generateMetadataTemplate({
  title = 'Cost Savvy Health',
  description = 'Find and compare healthcare costs across providers. Get transparent pricing for medical procedures and services.',
  keywords = ['healthcare', 'medical costs', 'health insurance', 'medical procedures', 'healthcare pricing'],
  image = DEFAULT_OG_IMAGE,
  url = BASE_URL,
}: MetadataTemplateProps = {}): Metadata {
  const normalizedTitle = normalizeBrand(title);
  const normalizedUrl = normalizeBrand(url);
  // Ensure image URL is absolute
  const imageUrl = image.startsWith('http') ? image : `${BASE_URL}${image}`;

  return {
    title: normalizedTitle,
    description,
    keywords: keywords.join(', '),
    openGraph: {
      title: normalizedTitle,
      description,
      url: normalizedUrl,
      siteName: 'Cost Savvy Health',
      images: [
        {
          url: imageUrl,
          width: 1200,
          height: 630,
          alt: normalizedTitle,
        },
      ],
      locale: 'en_US',
      type: 'website',
    },
    twitter: {
      card: 'summary_large_image',
      title: normalizedTitle,
      description,
      images: [imageUrl],
    },
    robots: {
      index: true,
      follow: true,
      googleBot: {
        index: true,
        follow: true,
        'max-video-preview': -1,
        'max-image-preview': 'large',
        'max-snippet': -1,
      },
    },
    verification: {
      google: 'your-google-site-verification', // Add your Google verification code
    },
  };
} 
