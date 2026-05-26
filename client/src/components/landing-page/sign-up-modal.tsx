"use client";

import React, { useState } from "react";
import { Building2, Hospital, Shield, User2, X } from "lucide-react";
import type { IntakeType } from "@/types/context/auth-user";

interface SignUpModalProps {
  onClose: () => void;
  onSelection: (option: IntakeType) => void;
}

const options: Array<{
  value: IntakeType;
  title: string;
  description: string;
  icon: React.ReactNode;
}> = [
  {
    value: "consumer",
    title: "Consumer / Patient",
    description:
      "Compare prices, find nearby hospitals, and understand care costs for yourself or your family.",
    icon: <User2 size={44} className="text-[#1B3B36]" />,
  },
  {
    value: "employer",
    title: "Employer",
    description:
      "Support employees with benefits education, cost navigation, and more transparent care choices.",
    icon: <Building2 size={44} className="text-[#1B3B36]" />,
  },
  {
    value: "provider",
    title: "Provider / Hospital",
    description:
      "Upload hospital data, manage CPT pricing records, and keep location and service details current.",
    icon: <Hospital size={44} className="text-[#1B3B36]" />,
  },
  {
    value: "payer",
    title: "Payer",
    description:
      "Manage network data, member transparency workflows, and pricing visibility for plan members.",
    icon: <Shield size={44} className="text-[#1B3B36]" />,
  },
];

export default function SignUpModal({ onClose, onSelection }: SignUpModalProps) {
  const [selected, setSelected] = useState<IntakeType | null>(null);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/20 backdrop-blur-sm">
      <div className="relative mx-4 w-full max-w-6xl rounded-3xl bg-white p-8 shadow-lg">
        <button
          onClick={onClose}
          className="absolute right-6 top-6 text-gray-400 transition-colors hover:text-gray-600"
          aria-label="Close modal"
        >
          <X size={24} />
        </button>

        <h1 className="mb-4 text-[2.5rem] font-semibold leading-tight text-[#1B3B36]">
          I want to use Cost Savvy as...
        </h1>
        <p className="mb-8 text-2xl text-gray-500">
          Choose the intake path that fits your organization.
        </p>

        <div className="mb-8 grid gap-6 md:grid-cols-2">
          {options.map((option) => {
            const isSelected = selected === option.value;
            return (
              <button
                key={option.value}
                onClick={() => setSelected(option.value)}
                className={`rounded-2xl border-2 p-6 text-left transition-all ${
                  isSelected
                    ? "border-[#1B3B36] bg-[#f5f9f8]"
                    : "border-gray-200 hover:border-gray-300"
                }`}
              >
                <div className="mb-4">{option.icon}</div>
                <h2 className="mb-2 text-2xl font-semibold text-[#1B3B36]">
                  {option.title}
                </h2>
                <p className="text-lg text-gray-500">{option.description}</p>
              </button>
            );
          })}
        </div>

        <button
          onClick={() => selected && onSelection(selected)}
          className={`w-full cursor-pointer rounded-full py-4 text-xl font-medium transition-all ${
            selected
              ? "bg-[#8C2F5D] text-white hover:bg-[#C85990]"
              : "cursor-not-allowed bg-gray-100 text-gray-400"
          }`}
          disabled={!selected}
        >
          Continue
        </button>
      </div>
    </div>
  );
}
