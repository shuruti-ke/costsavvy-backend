"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { CheckCircle2, Mail, ShieldAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

type ConfirmationState =
  | { status: "loading" }
  | { status: "success"; dashboardPath: string; accountType: string | null; message: string }
  | { status: "error"; message: string };

export default function EmailConfirmationView({ token }: { token: string }) {
  const [state, setState] = useState<ConfirmationState>({ status: "loading" });

  const signInHref = useMemo(() => {
    const accountType = state.status === "success" ? state.accountType : null;
    return accountType === "business" ? "/auth?type=employer" : "/auth?type=consumer";
  }, [state]);

  useEffect(() => {
    if (!token) {
      setState({ status: "error", message: "Missing confirmation token." });
      return;
    }

    let cancelled = false;
    const confirm = async () => {
      try {
        const response = await fetch(`/api/auth/confirm?token=${encodeURIComponent(token)}`);
        const data = await response.json();
        if (cancelled) return;
        if (!response.ok) {
          setState({
            status: "error",
            message: data?.message || "We could not confirm your account.",
          });
          return;
        }

        setState({
          status: "success",
          dashboardPath: data?.dashboardPath || "/",
          accountType: data?.accountType || null,
          message: data?.message || "Your email has been confirmed.",
        });
      } catch (error) {
        if (cancelled) return;
        setState({
          status: "error",
          message:
            error instanceof Error
              ? error.message
              : "We could not confirm your account.",
        });
      }
    };

    confirm();

    return () => {
      cancelled = true;
    };
  }, [token]);

  return (
    <div className="min-h-svh bg-[radial-gradient(circle_at_top,_rgba(200,89,144,0.24),_transparent_30%),linear-gradient(180deg,#120710_0%,#090509_100%)] px-6 py-16 text-white">
      <div className="mx-auto flex max-w-3xl items-center justify-center">
        <Card className="w-full border-white/10 bg-white/95 text-gray-900 shadow-2xl shadow-black/30">
          <CardHeader className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-[#8C2F5D]/10 text-[#8C2F5D]">
              {state.status === "success" ? (
                <CheckCircle2 className="h-8 w-8" />
              ) : state.status === "error" ? (
                <ShieldAlert className="h-8 w-8" />
              ) : (
                <Mail className="h-8 w-8 animate-pulse" />
              )}
            </div>
            <CardTitle className="text-3xl">
              {state.status === "success"
                ? "Email confirmed"
                : state.status === "error"
                  ? "Confirmation failed"
                  : "Confirming your email"}
            </CardTitle>
            <CardDescription className="text-base text-gray-600">
              {state.status === "success"
                ? "Your account is now active and ready to use."
                : state.status === "error"
                  ? "We could not activate this account from the link you opened."
                  : "We’re verifying your account now."}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 px-6 pb-8">
            {state.status === "loading" && (
              <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                Please wait while we confirm your account.
              </div>
            )}

            {state.status === "success" && (
              <div className="rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-4 text-sm text-emerald-900">
                <p className="font-semibold">{state.message}</p>
                <p className="mt-2 leading-6">
                  {state.accountType === "business"
                    ? "Business accounts can use the platform dashboard to upload hospital data, review pricing content, and manage updates."
                    : "You can now sign in and use the consumer experience."}
                </p>
              </div>
            )}

            {state.status === "error" && (
              <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-4 text-sm text-rose-900">
                <p className="font-semibold">{state.message}</p>
                <p className="mt-2 leading-6">
                  If the link expired, return to the sign in page and request a
                  new confirmation email.
                </p>
              </div>
            )}

            <div className="flex flex-wrap gap-3">
              {state.status === "success" ? (
                <>
                  <Button asChild className="rounded-full bg-[#8C2F5D] text-white hover:bg-[#C85990]">
                    <Link href={state.dashboardPath}>
                      {state.accountType === "business"
                        ? "Go to business dashboard"
                        : "Go to home"}
                    </Link>
                  </Button>
                  <Button asChild variant="outline" className="rounded-full border-gray-300">
                    <Link href={signInHref}>Sign in now</Link>
                  </Button>
                </>
              ) : (
                <>
                  <Button asChild className="rounded-full bg-[#8C2F5D] text-white hover:bg-[#C85990]">
                    <Link href={signInHref}>Back to sign in</Link>
                  </Button>
                  <Button asChild variant="outline" className="rounded-full border-gray-300">
                    <Link href="/">Back to site</Link>
                  </Button>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
