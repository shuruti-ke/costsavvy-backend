"use client";

import React, { FormEvent, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { ChevronRight, CheckCircle2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { useAuth } from "@/context/AuthContext";
import { googleAuthUrl } from "@/api/auth/api";
import { toast } from "sonner";
import type { IntakeType } from "@/types/context/auth-user";

function getIntakeCopy(type: IntakeType) {
  switch (type) {
    case "employer":
      return {
        title: "Employer onboarding",
        description:
          "Tell us how many employees you support and what you want to improve across benefits, pricing, or navigation.",
        companyLabel: "Employer / company name",
        roleLabel: "Your role",
        sizeLabel: "Employee count",
        goalLabel: "What are you trying to improve?",
        goalPlaceholder:
          "Example: help employees compare prices, reduce plan confusion, or improve benefit engagement.",
      };
    case "provider":
      return {
        title: "Provider onboarding",
        description:
          "Share your organization details so we can tailor the platform to hospital, practice, or system workflows.",
        companyLabel: "Hospital / practice name",
        roleLabel: "Your title",
        sizeLabel: "Locations / facilities",
        goalLabel: "How do you plan to use Cost Savvy Health?",
        goalPlaceholder:
          "Example: publish better pricing data, manage CPT records, or improve patient cost transparency.",
      };
    case "payer":
      return {
        title: "Payer onboarding",
        description:
          "Add enough context for your health plan or payer team so we can align the platform to network and benefit use cases.",
        companyLabel: "Plan / payer name",
        roleLabel: "Your title",
        sizeLabel: "Covered lives",
        goalLabel: "What are you trying to manage?",
        goalPlaceholder:
          "Example: improve member transparency, compare rates, or support network strategy.",
      };
    default:
      return {
        title: "Consumer onboarding",
        description:
          "Create your personal account to compare prices and find care near you.",
        companyLabel: "Organization name (optional)",
        roleLabel: "Your role (optional)",
        sizeLabel: "Household size (optional)",
        goalLabel: "What are you trying to do?",
        goalPlaceholder:
          "Example: compare CPT prices, find nearby hospitals, or check insurance options.",
      };
  }
}

export default function SignUpForm({
  authType,
  onSwitch,
}: {
  authType: IntakeType | null;
  onSwitch: () => void;
}) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { register, isAuthenticated, isLoading } = useAuth();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [verificationEmail, setVerificationEmail] = useState<string | null>(null);
  const [dashboardPath, setDashboardPath] = useState<string | null>(null);
  const [intakeType, setIntakeType] = useState<IntakeType>(
    authType || "consumer"
  );

  const from = searchParams.get("from") || "/";
  const copy = useMemo(() => getIntakeCopy(intakeType), [intakeType]);

  useEffect(() => {
    if (isAuthenticated && !isLoading) {
      router.push(from);
    }
  }, [isAuthenticated, isLoading, router, from]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsSubmitting(true);
    setError(null);

    const formData = new FormData(event.currentTarget);
    const name = String(formData.get("name") || "").trim();
    const email = String(formData.get("email") || "").trim().toLowerCase();
    const password = String(formData.get("password") || "");
    const phoneNumber = String(formData.get("phoneNumber") || "").trim();
    const companyName = String(formData.get("companyName") || "").trim();
    const jobTitle = String(formData.get("jobTitle") || "").trim();
    const organizationSize = String(formData.get("organizationSize") || "").trim();
    const specialty = String(formData.get("specialty") || "").trim();
    const primaryGoal = String(formData.get("primaryGoal") || "").trim();
    const zipCode = String(formData.get("zipCode") || "").trim();
    const useCase = String(formData.get("useCase") || "").trim();
    const accountType = intakeType === "consumer" ? "consumer" : "business";

    if (intakeType !== "consumer" && !companyName) {
      setError("Please add your organization or company name.");
      setIsSubmitting(false);
      return;
    }

    try {
      const response = await register({
        name,
        email,
        password,
        phoneNumber,
        companyName: companyName || undefined,
        jobTitle: jobTitle || undefined,
        organizationType: intakeType,
        zipCode,
        organizationSize: organizationSize || undefined,
        specialty: specialty || undefined,
        primaryGoal: primaryGoal || undefined,
        useCase: useCase || primaryGoal || undefined,
        accountType,
      });

      if (response.confirmationRequired) {
        setVerificationEmail(email);
        setDashboardPath(response.dashboardPath || null);
        toast.success(response.message || "Check your email to verify your account.");
      } else {
        toast.success(response.message || "Registration successful");
      }
    } catch (err) {
      console.error("Registration error:", err);
      setError(
        err instanceof Error
          ? err.message
          : "Failed to register. Please try again."
      );
      toast.error("Registration failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleGoogleSignUp = () => {
    localStorage.setItem("authRedirectUrl", from);
    window.location.href = googleAuthUrl;
  };

  return (
    <Card className="bg-white border border-gray-300 text-gray-900 rounded-t-lg pt-0">
      <Link
        href="/"
        className="flex items-center justify-center bg-gray-100 p-2 rounded-t-lg"
      >
        <img src="/icon-black.png" alt="" />
      </Link>

      <div className="text-center text-normal text-gray-600 flex items-center justify-center border-b pb-3 gap-16">
        <button
          type="button"
          onClick={onSwitch}
          className="hover:underline underline-offset-4 hover:text-gray-800 cursor-pointer"
        >
          Sign in
        </button>
        <button
          type="button"
          className="font-medium text-gray-800 underline underline-offset-4 cursor-pointer"
        >
          Sign up
        </button>
      </div>

      <CardHeader className="text-center pb-2">
        <CardTitle className="text-xl font-bold">
          {verificationEmail ? "Check your inbox" : copy.title}
        </CardTitle>
        <CardDescription className="text-sm text-gray-600">
          {verificationEmail
            ? `We sent a confirmation link to ${verificationEmail}. Open it to verify your account, then sign in${dashboardPath ? ` to reach ${dashboardPath}` : ""}.`
            : copy.description}
        </CardDescription>
      </CardHeader>

      <CardContent>
        {error && (
          <div className="bg-red-50 text-red-500 p-2 mb-4 rounded-md text-sm">
            {error}
          </div>
        )}

        {verificationEmail ? (
          <div className="space-y-4 rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-900">
            <div className="flex items-start gap-3">
              <CheckCircle2 className="mt-0.5 h-5 w-5 text-emerald-600" />
              <div>
                <p className="font-semibold">Verification email sent</p>
                <p className="mt-1 leading-6">
                  Confirm your email to activate your account. If you do not see
                  the message within a few minutes, check spam or promotions.
                </p>
                {dashboardPath && (
                  <p className="mt-2 text-emerald-800">
                    Business accounts can sign in at {dashboardPath} after verification.
                  </p>
                )}
              </div>
            </div>
            <div className="flex flex-wrap gap-3">
              <Button
                type="button"
                className="rounded-full bg-[#8C2F5D] text-white hover:bg-[#C85990]"
                onClick={onSwitch}
              >
                Go to sign in
              </Button>
              <Button
                type="button"
                variant="outline"
                className="rounded-full border-gray-300 bg-white text-gray-900 hover:bg-gray-50"
                onClick={() => {
                  setVerificationEmail(null);
                  setDashboardPath(null);
                }}
              >
                Create another account
              </Button>
            </div>
          </div>
        ) : (
          <form onSubmit={handleSubmit}>
            <div className="grid gap-4">
              <div className="grid gap-1">
                <Label htmlFor="signup-path">Intake path</Label>
                <Select
                  value={intakeType}
                  onValueChange={(value) => setIntakeType(value as IntakeType)}
                  disabled={isSubmitting}
                >
                  <SelectTrigger
                    id="signup-path"
                    className="w-full border border-gray-300 bg-white text-gray-900 h-9"
                  >
                    <SelectValue placeholder="Select a path" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="consumer">Consumer / Patient</SelectItem>
                    <SelectItem value="employer">Employer</SelectItem>
                    <SelectItem value="provider">Provider / Hospital</SelectItem>
                    <SelectItem value="payer">Payer</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="grid gap-1">
                <Label htmlFor="signup-name">Full name</Label>
                <Input
                  id="signup-name"
                  name="name"
                  type="text"
                  required
                  className="bg-white border border-gray-300 text-gray-900 placeholder:text-gray-800 focus-visible:ring-gray-300 h-9"
                  disabled={isSubmitting}
                />
              </div>

              <div className="grid gap-1">
                <Label htmlFor="signup-email">Email</Label>
                <Input
                  id="signup-email"
                  name="email"
                  type="email"
                  required
                  className="bg-white border border-gray-300 text-gray-900 placeholder:text-gray-800 focus-visible:ring-gray-300 h-9"
                  disabled={isSubmitting}
                />
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <div className="grid gap-1">
                  <Label htmlFor="signup-phone">Phone number</Label>
                  <Input
                    id="signup-phone"
                    name="phoneNumber"
                    type="tel"
                    required
                    placeholder="(555) 555-5555"
                    className="bg-white border border-gray-300 text-gray-900 placeholder:text-gray-500 focus-visible:ring-gray-300 h-9"
                    disabled={isSubmitting}
                  />
                </div>

                <div className="grid gap-1">
                  <Label htmlFor="signup-zip">ZIP code</Label>
                  <Input
                    id="signup-zip"
                    name="zipCode"
                    type="text"
                    required
                    placeholder="06119"
                    className="bg-white border border-gray-300 text-gray-900 placeholder:text-gray-500 focus-visible:ring-gray-300 h-9"
                    disabled={isSubmitting}
                  />
                </div>
              </div>

              <div className="grid gap-1">
                <Label htmlFor="signup-company">{copy.companyLabel}</Label>
                <Input
                  id="signup-company"
                  name="companyName"
                  type="text"
                  required={intakeType !== "consumer"}
                  className="bg-white border border-gray-300 text-gray-900 placeholder:text-gray-500 focus-visible:ring-gray-300 h-9"
                  disabled={isSubmitting}
                />
              </div>

              <div className="grid gap-1">
                <Label htmlFor="signup-role">{copy.roleLabel}</Label>
                <Input
                  id="signup-role"
                  name="jobTitle"
                  type="text"
                  required={intakeType !== "consumer"}
                  className="bg-white border border-gray-300 text-gray-900 placeholder:text-gray-500 focus-visible:ring-gray-300 h-9"
                  disabled={isSubmitting}
                />
              </div>

              {intakeType !== "consumer" && (
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="grid gap-1">
                    <Label htmlFor="signup-size">{copy.sizeLabel}</Label>
                    <Input
                      id="signup-size"
                      name="organizationSize"
                      type="text"
                      placeholder={
                        intakeType === "employer"
                          ? "500 employees"
                          : intakeType === "payer"
                            ? "250,000 covered lives"
                            : "10 locations"
                      }
                      className="bg-white border border-gray-300 text-gray-900 placeholder:text-gray-500 focus-visible:ring-gray-300 h-9"
                      disabled={isSubmitting}
                    />
                  </div>

                  <div className="grid gap-1">
                    <Label htmlFor="signup-specialty">
                      {intakeType === "provider"
                        ? "Specialty / service line"
                        : intakeType === "payer"
                          ? "Line of business"
                          : "Industry / focus"}
                    </Label>
                    <Input
                      id="signup-specialty"
                      name="specialty"
                      type="text"
                      placeholder={
                        intakeType === "provider"
                          ? "Radiology, primary care, surgery"
                          : intakeType === "payer"
                            ? "Commercial, Medicaid, Medicare Advantage"
                            : "Healthcare, benefits, HR"
                      }
                      className="bg-white border border-gray-300 text-gray-900 placeholder:text-gray-500 focus-visible:ring-gray-300 h-9"
                      disabled={isSubmitting}
                    />
                  </div>
                </div>
              )}

              <div className="grid gap-1">
                <Label htmlFor="signup-goal">{copy.goalLabel}</Label>
                <Textarea
                  id="signup-goal"
                  name="primaryGoal"
                  required
                  rows={4}
                  placeholder={copy.goalPlaceholder}
                  className="bg-white border border-gray-300 text-gray-900 placeholder:text-gray-500 focus-visible:ring-gray-300"
                  disabled={isSubmitting}
                />
              </div>

              <div className="grid gap-1">
                <Label htmlFor="signup-password">Password</Label>
                <Input
                  id="signup-password"
                  name="password"
                  type="password"
                  required
                  className="bg-white border border-gray-300 text-gray-900 placeholder:text-gray-800 focus-visible:ring-gray-300 h-9"
                  disabled={isSubmitting}
                />
              </div>

              <div className="rounded-md bg-rose-50 px-3 py-2 text-xs leading-5 text-gray-700">
                After signup, we will email you a verification link before the
                account becomes active.
              </div>

              <Button
                type="submit"
                className="w-full bg-[#8C2F5D] hover:bg-[#C85990] text-white text-sm font-medium uppercase flex items-center justify-center gap-2 h-9 cursor-pointer"
                disabled={isSubmitting}
              >
                {isSubmitting ? (
                  <span className="flex items-center justify-center">
                    <svg
                      className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Creating account...
                  </span>
                ) : (
                  <>
                    Create account
                    <ChevronRight className="h-4 w-4" />
                  </>
                )}
              </Button>
            </div>

            <div className="mt-4">
              <Button
                type="button"
                variant="outline"
                className="w-full gap-2 bg-white hover:bg-gray-50 border border-gray-400 text-gray-900 text-sm cursor-pointer"
                onClick={handleGoogleSignUp}
              >
                Continue with Google
              </Button>
            </div>
          </form>
        )}
      </CardContent>
    </Card>
  );
}
