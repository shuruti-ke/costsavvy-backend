"use client";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { useSearchParams } from "next/navigation";
import SignInForm from "./sign-in-form";
import SignUpForm from "./sign-up-form";
import type { IntakeType } from "@/types/context/auth-user";

export function AuthSlider() {
  const searchParams = useSearchParams();
  const rawAuthType = searchParams.get("type");
  const authType: IntakeType | null =
    rawAuthType === "business" || rawAuthType === "employer"
      ? "employer"
      : rawAuthType === "consumer" ||
          rawAuthType === "provider" ||
          rawAuthType === "payer"
        ? rawAuthType
        : null;

  const [selectedForm, setSelectedForm] = useState<"signin" | "signup">(
    authType ? "signup" : "signin"
  );

  useEffect(() => {
    if (authType) {
      setSelectedForm("signup");
    }
  }, [authType]);

  return (
    <div className="relative flex w-full items-start justify-center overflow-x-hidden overflow-y-visible min-h-[1040px] pt-2">
      <div
        className={cn(
          "flex w-[200%] absolute left-0 transition-all duration-300 ease-in-out",
          {
            "translate-x-0": selectedForm === "signin",
            "-translate-x-1/2": selectedForm === "signup",
          }
        )}
      >
        <div className="w-1/2 px-4">
          <SignInForm
            authType={authType}
            onSwitch={() => selectedForm === "signin" && setSelectedForm("signup")}
          />
        </div>
        <div className="w-1/2 px-4">
          <SignUpForm
            authType={authType}
            onSwitch={() => selectedForm === "signup" && setSelectedForm("signin")}
          />
        </div>
      </div>
    </div>
  );
}
