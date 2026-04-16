"use client";

import { FormEvent, useMemo, useState, type ReactNode } from "react";
import Link from "next/link";
import { ArrowUpRight, Database, FileUp, Hospital, LogOut, ShieldCheck, Users } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import SignInForm from "@/components/auth/sign-in-form";
import { useAuth } from "@/context/AuthContext";
import { toast } from "sonner";

function StatCard({
  title,
  value,
  description,
  icon,
}: {
  title: string;
  value: string;
  description: string;
  icon: ReactNode;
}) {
  return (
    <Card className="border-white/10 bg-white/5 text-white shadow-xl shadow-black/20">
      <CardContent className="flex items-start justify-between gap-4 p-5">
        <div>
          <p className="text-xs uppercase tracking-[0.24em] text-white/60">{title}</p>
          <p className="mt-3 text-3xl font-semibold">{value}</p>
          <p className="mt-2 text-sm text-white/70">{description}</p>
        </div>
        <div className="rounded-full border border-white/10 bg-white/10 p-3 text-white">
          {icon}
        </div>
      </CardContent>
    </Card>
  );
}

function ActionCard({
  title,
  description,
  href,
  icon,
  external = false,
}: {
  title: string;
  description: string;
  href: string;
  icon: ReactNode;
  external?: boolean;
}) {
  return (
    <Card className="border-white/10 bg-white/5 text-white shadow-xl shadow-black/20">
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-3">
            <div className="inline-flex rounded-full border border-white/10 bg-white/10 p-2 text-white">
              {icon}
            </div>
            <div>
              <p className="text-lg font-semibold">{title}</p>
              <p className="mt-1 max-w-sm text-sm text-white/70">{description}</p>
            </div>
          </div>
          <Button
            asChild
            variant="outline"
            className="shrink-0 border-white/15 bg-white/10 text-white hover:bg-white/15"
          >
            {external ? (
              <a href={href} target="_blank" rel="noreferrer">
                Open
                <ArrowUpRight className="h-4 w-4" />
              </a>
            ) : (
              <Link href={href}>
                Open
                <ArrowUpRight className="h-4 w-4" />
              </Link>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default function BusinessDashboard() {
  const { user, isLoading, logout } = useAuth();
  const [submitting, setSubmitting] = useState(false);
  const [attachmentName, setAttachmentName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const isBusiness = user?.role === "admin" || user?.accountType === "business";

  const summary = useMemo(
    () => ({
      organization: user?.organizationType || "Business",
      company: user?.companyName || "Your organization",
      role: user?.jobTitle || "Business user",
    }),
    [user]
  );

  const handleLogout = async () => {
    await logout();
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setSuccess(null);

    const formData = new FormData(event.currentTarget);

    try {
      const response = await fetch("/api/business/intake", {
        method: "POST",
        credentials: "include",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.message || "Failed to send intake request");
      }

      setSuccess(data?.message || "Request sent");
      toast.success("Business data request sent");
      event.currentTarget.reset();
      setAttachmentName(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to send intake request";
      setError(message);
      toast.error(message);
    } finally {
      setSubmitting(false);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-svh bg-[radial-gradient(circle_at_top,_rgba(200,89,144,0.25),_transparent_28%),linear-gradient(180deg,#14070f_0%,#090509_100%)] px-6 py-16 text-white">
        <div className="mx-auto flex max-w-6xl items-center justify-center">
          <Card className="w-full max-w-xl border-white/10 bg-white/5 text-white shadow-2xl shadow-black/30">
            <CardContent className="p-8 text-center">
              <div className="mx-auto mb-4 h-12 w-12 animate-spin rounded-full border-2 border-white/20 border-t-white/80" />
              <p className="text-lg font-semibold">Checking business access</p>
              <p className="mt-2 text-sm text-white/70">
                Verifying your session before loading the dashboard.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-svh bg-[radial-gradient(circle_at_top,_rgba(200,89,144,0.25),_transparent_28%),linear-gradient(180deg,#14070f_0%,#090509_100%)] px-6 py-16 text-white">
        <div className="mx-auto grid max-w-6xl gap-10 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="space-y-6">
            <div className="space-y-4">
              <p className="inline-flex rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold uppercase tracking-[0.24em] text-white/70">
                Business dashboard
              </p>
              <h1 className="max-w-3xl font-serif text-4xl leading-tight text-white md:text-5xl">
                Sign in to manage your organization&apos;s data.
              </h1>
              <p className="max-w-2xl text-base leading-7 text-white/70">
                Upload hospital files, keep CPT records current, and review the
                business-specific workflow from one dashboard.
              </p>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <StatCard
                title="Account"
                value="Business only"
                description="Protected sign-in for providers, payers, and employers."
                icon={<Users className="h-5 w-5" />}
              />
              <StatCard
                title="Data"
                value="Uploads"
                description="Submit hospital or CPT updates for review."
                icon={<FileUp className="h-5 w-5" />}
              />
            </div>

            <Card className="border-white/10 bg-white/5 text-white shadow-2xl shadow-black/20">
              <CardContent className="p-6">
                <div className="flex flex-wrap items-center gap-3 text-sm text-white/70">
                  <ShieldCheck className="h-5 w-5 text-[#FFD6E9]" />
                  <span>
                    Business access is only available after email verification.
                  </span>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="lg:pt-10">
            <SignInForm
              authType={null}
              onSwitch={() => undefined}
              redirectTo="/dashboard/business"
              hideSwitch
            />
          </div>
        </div>
      </div>
    );
  }

  if (!isBusiness) {
    return (
      <div className="min-h-svh bg-[radial-gradient(circle_at_top,_rgba(200,89,144,0.25),_transparent_28%),linear-gradient(180deg,#14070f_0%,#090509_100%)] px-6 py-16 text-white">
        <div className="mx-auto flex max-w-3xl items-center justify-center">
          <Card className="w-full border-white/10 bg-white/5 text-white shadow-2xl shadow-black/30">
            <CardHeader>
              <CardTitle className="text-2xl">Business access required</CardTitle>
              <CardDescription className="text-white/70">
                The account currently signed in is not a business account.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-white/70">
                Please sign out and log back in with an employer, provider, or
                payer account to manage uploads and organization data.
              </p>
              <div className="flex flex-wrap gap-3">
                <Button
                  onClick={handleLogout}
                  className="rounded-full bg-[#C85990] px-5 text-white hover:bg-[#D56AA0]"
                >
                  <LogOut className="h-4 w-4" />
                  Sign out
                </Button>
                <Button asChild variant="outline" className="rounded-full border-white/15 bg-white/10 text-white hover:bg-white/15">
                  <Link href="/">Back to site</Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-svh bg-[radial-gradient(circle_at_top,_rgba(200,89,144,0.25),_transparent_28%),linear-gradient(180deg,#14070f_0%,#090509_100%)] px-6 py-10 text-white">
      <div className="mx-auto max-w-7xl space-y-8">
        <div className="flex flex-col justify-between gap-6 rounded-[28px] border border-white/10 bg-white/5 p-8 shadow-2xl shadow-black/30 lg:flex-row lg:items-end">
          <div className="max-w-3xl space-y-4">
            <p className="inline-flex rounded-full border border-white/10 bg-white/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.24em] text-white/70">
              Business Dashboard
            </p>
            <h1 className="font-serif text-4xl leading-tight md:text-5xl">
              Welcome back, {user?.name || "business partner"}.
            </h1>
            <p className="max-w-2xl text-base leading-7 text-white/70">
              Manage facility uploads, CPT lists, and organization updates from
              your private dashboard.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <Button asChild variant="outline" className="rounded-full border-white/15 bg-white/10 text-white hover:bg-white/15">
              <Link href="/auth?type=employer">
                Sign in again
                <ArrowUpRight className="h-4 w-4" />
              </Link>
            </Button>
            <Button
              onClick={handleLogout}
              className="rounded-full bg-[#C85990] px-5 text-white hover:bg-[#D56AA0]"
            >
              <LogOut className="h-4 w-4" />
              Sign out
            </Button>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <StatCard
            title="Organization"
            value={summary.organization}
            description="Your account intake path."
            icon={<Hospital className="h-5 w-5" />}
          />
          <StatCard
            title="Company"
            value={summary.company}
            description="The organization tied to this account."
            icon={<Users className="h-5 w-5" />}
          />
          <StatCard
            title="Role"
            value={summary.role}
            description="Your current job title or focus."
            icon={<ShieldCheck className="h-5 w-5" />}
          />
          <StatCard
            title="Data"
            value="Postgres"
            description="Uploads and platform records are linked to the backend."
            icon={<Database className="h-5 w-5" />}
          />
        </div>

        <div className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
          <Card className="border-white/10 bg-white/5 text-white shadow-2xl shadow-black/20">
            <CardHeader className="border-b border-white/10">
              <CardTitle className="text-2xl">Upload hospital or CPT data</CardTitle>
              <CardDescription className="text-white/70">
                Send a CSV or notes file to the Cost Savvy Health team for review.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 p-6">
              {error && (
                <div className="rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
                  {error}
                </div>
              )}
              {success && (
                <div className="rounded-2xl border border-emerald-400/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-100">
                  {success}
                </div>
              )}

              <form onSubmit={handleSubmit} className="space-y-4">
                <input type="hidden" name="contactName" value={user?.name || ""} />
                <input type="hidden" name="email" value={user?.email || ""} />
                <input type="hidden" name="companyName" value={user?.companyName || ""} />
                <input type="hidden" name="organizationType" value={user?.organizationType || ""} />

                <div className="grid gap-4 md:grid-cols-2">
                  <label className="grid gap-2 text-sm">
                    <span>Upload type</span>
                    <select
                      name="uploadType"
                      className="rounded-xl border border-white/15 bg-white/10 px-3 py-2 text-white outline-none"
                      required
                    >
                      <option value="Hospital data">Hospital data</option>
                      <option value="CPT updates">CPT updates</option>
                      <option value="Payer network">Payer network</option>
                      <option value="Other">Other</option>
                    </select>
                  </label>
                  <label className="grid gap-2 text-sm">
                    <span>Attach file</span>
                    <input
                      type="file"
                      name="file"
                      accept=".csv,.xlsx,.xls,.pdf,.txt"
                      onChange={(event) => setAttachmentName(event.target.files?.[0]?.name || null)}
                      className="rounded-xl border border-white/15 bg-white/10 px-3 py-2 text-white file:mr-4 file:rounded-full file:border-0 file:bg-white file:px-4 file:py-2 file:text-sm file:font-medium file:text-gray-900"
                    />
                    {attachmentName && (
                      <span className="text-xs text-white/60">{attachmentName}</span>
                    )}
                  </label>
                </div>

                <label className="grid gap-2 text-sm">
                  <span>Notes</span>
                  <textarea
                    name="notes"
                    rows={5}
                    className="rounded-2xl border border-white/15 bg-white/10 px-4 py-3 text-white outline-none placeholder:text-white/40"
                    placeholder="Tell us what this file contains, which facilities or services it covers, and any special handling notes."
                  />
                </label>

                <Button
                  type="submit"
                  disabled={submitting}
                  className="rounded-full bg-[#C85990] px-5 text-white hover:bg-[#D56AA0]"
                >
                  {submitting ? "Sending..." : "Send data intake"}
                </Button>
              </form>
            </CardContent>
          </Card>

          <div className="space-y-6">
            <ActionCard
              title="Review CPT management"
              description="Open the pricing workflow to review how procedure data appears to members."
              href="/"
              icon={<FileUp className="h-5 w-5" />}
            />
            <ActionCard
              title="Content updates"
              description="Ask the platform team to publish new articles, pages, or site copy in Sanity."
              href="https://cost-savy.sanity.studio/structure"
              icon={<Database className="h-5 w-5" />}
              external
            />
            <ActionCard
              title="Need help?"
              description="Reach the platform team if you need a deeper integration or bulk import flow."
              href="mailto:chat@costsavvy.health"
              icon={<ShieldCheck className="h-5 w-5" />}
              external
            />
          </div>
        </div>
      </div>
    </div>
  );
}
