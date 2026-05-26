"use client";
import MobileMenu from "./mobile-menu";
import React, { useState } from "react";
import NavLinks from "./nav-links";
import Hamburger from "./hamburger-icon";
import { X } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import SignUpModal from "./landing-page/sign-up-modal";
import type { IntakeType } from "@/types/context/auth-user";
export default function Navbar() {
  // STATES
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // HANDLERS
  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const openModal = () => setIsModalOpen(true);
  const closeModal = () => setIsModalOpen(false);
  const router = useRouter();
  const handleSelection = (option: IntakeType) => {
    router.push(`/auth?type=${option}`);
    if (option) closeModal();
  };

  return (
    <div>
      <nav className="py-3 px-2 md:px-5  bg-[#6B1548] w-full">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Link href="/">
              <img src="/logo.png" alt="Cost Savvy Health" className="w-[80%]" />
            </Link>
          </div>

          <div className="hidden lg:flex items-center gap-3 text-[16px]">
            <NavLinks />
            <div className="mx-1 h-6 w-px bg-white/30" aria-hidden="true" />

            <button
              onClick={openModal}
              className="inline-flex items-center rounded-full border border-white/20 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-white/10 hover:border-white/30"
            >
              Sign Up
            </button>
            <Link href="/admin" className="inline-flex">
              <span className="inline-flex items-center rounded-full bg-[#A34E78] px-4 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-[#B85B84]">
                Platform Sign In
              </span>
            </Link>
          </div>

          <div className="lg:hidden">
            {isMenuOpen ? (
              <button onClick={toggleMenu} className="text-white">
                <X size={30} />
              </button>
            ) : (
              <Hamburger isOpen={isMenuOpen} toggleMenu={toggleMenu} />
            )}
          </div>
        </div>
      </nav>

      <MobileMenu isMenuOpen={isMenuOpen} toggleMenu={toggleMenu} />

      {isModalOpen && (
        <SignUpModal onClose={closeModal} onSelection={handleSelection} />
      )}
    </div>
  );
}
