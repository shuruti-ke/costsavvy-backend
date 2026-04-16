import { generateMetadataTemplate } from '@/lib/metadata';
import { Metadata } from 'next';

export async function generateMetadata(): Promise<Metadata> {
  return generateMetadataTemplate({
    title: 'Home | Cost Savvy Health',
    description: 'Learn about Cost Savvy Health\'s mission to bring transparency to healthcare costs and help patients make informed decisions about their medical care.',
    keywords: [
      'Home | Cost Savy Health',
      'healthcare transparency',
      'medical cost transparency',
      'healthcare mission',
      'healthcare cost comparison',
      'patient advocacy'
    ],
    url: 'https://www.costsavvy.health/',
  });
}

export const dynamic = 'force-dynamic'  
export const fetchCache = 'force-no-store'

import React from 'react'
import { getHomePage } from '@/api/sanity/queries'
import Hero from '@/components/landing-page/hero'
import FeatureCards from '@/components/features-card'
import ShopHealthcare from '@/components/landing-page/shop-health-care'
import PriceTransparency from '@/components/landing-page/price-transparency'
import Testimonial from '@/components/testimonial'
import Enterprise from '@/components/landing-page/enterprise'
import EnterpriseSolutions from '@/components/enterprise-solution'
import JoinTeam from '@/components/landing-page/join-team'

const fallbackHomeData = {
  hero: {
    tagline: "Know what you'll pay",
    rotatingWords: ['CT scans', 'MRIs', 'X-rays'],
    commonProcedures: ['MRI', 'CT scan', 'X-ray'],
  },
  featureCards: { cards: [] },
  shopHealthcare: {
    heading: 'Shop healthcare with confidence',
    description: 'Compare common services near you and understand your options before you book.',
    iconImage: '/icon.png',
    services: { sectionTitle: '', items: [] },
  },
  priceTransparency: {
    heading: 'Price transparency made simple',
    description: 'Search for estimated costs and discover nearby care options.',
    ctaText: 'Start searching',
    ctaLink: '/quote',
    features: [],
  },
  testimonial: {
    testimonial: 'Healthcare should be understandable, comparable, and transparent.',
    image: '/icon.png',
  },
  enterprise: {
    heading: 'Built for healthcare teams',
    description: 'Support patients, employers, and partners with better pricing visibility.',
    ctaText: 'Learn more',
    ctaLink: '/contact-us',
    iconImage: '/icon.png',
    features: [],
  },
  enterpriseSolutions: {
    solutions: [],
  },
  joinTeam: {
    heading: 'Stay in the loop',
    description: 'Read the latest updates and articles from the Cost Savvy Health team.',
    ctaText: 'Read articles',
    image: '/icon.png',
  },
} as const;

export default async function HomePage() {
  const homeData = await getHomePage().catch(() => null)
  const data = homeData ?? fallbackHomeData
  return (
    <>
      <Hero {...data.hero} />
      <FeatureCards cards={data.featureCards.cards} />
      <ShopHealthcare {...data.shopHealthcare} />
      <PriceTransparency {...data.priceTransparency} />
      <Testimonial
        testimonial={data.testimonial.testimonial}
        image={data.testimonial.image}
      />
      <Enterprise {...data.enterprise} />
      <EnterpriseSolutions solutions={data.enterpriseSolutions.solutions} />
      <JoinTeam {...data.joinTeam} />
    </>
  )
}
