"use client";

import React, { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowUpRight,
  Database,
  FileText,
  CheckCircle2,
  Hospital,
  LogOut,
  MessageSquare,
  ShieldCheck,
  XCircle,
  Users,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import SignInForm from "@/components/auth/sign-in-form";
import { useAuth } from "@/context/AuthContext";
import {
  approveSearchLearningReview,
  getAllUsers,
  getSearchLearningAliases,
  getSearchLearningReviews,
  rejectSearchLearningReview,
  type SearchLearningAlias,
  type SearchLearningReview,
} from "@/api/auth/api";
import type { User } from "@/types/context/auth-user";
import { toast } from "sonner";

const SANITY_STUDIO_URL = "https://cost-savy.sanity.studio/structure";

function StatCard({
  title,
  value,
  description,
  icon,
  tone = "default",
}: {
  title: string;
  value: string;
  description: string;
  icon: React.ReactNode;
  tone?: "default" | "violet" | "sky" | "emerald";
}) {
  const toneClasses =
    tone === "violet"
      ? "bg-[linear-gradient(135deg,rgba(200,89,144,0.18),rgba(255,255,255,0.02))]"
      : tone === "sky"
        ? "bg-[linear-gradient(135deg,rgba(88,160,255,0.18),rgba(255,255,255,0.02))]"
        : tone === "emerald"
          ? "bg-[linear-gradient(135deg,rgba(16,185,129,0.18),rgba(255,255,255,0.02))]"
          : "bg-[linear-gradient(135deg,rgba(255,255,255,0.08),rgba(255,255,255,0.02))]";

  return (
    <Card className={`overflow-hidden border-white/10 bg-white/5 text-white shadow-2xl shadow-black/20 ${toneClasses}`}>
      <CardContent className="flex items-start justify-between gap-4 p-5">
        <div className="space-y-3">
          <div className="inline-flex rounded-full border border-white/10 bg-black/20 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-white/70">
            {title}
          </div>
          <div className="text-3xl font-semibold tracking-tight">{value}</div>
          <p className="max-w-sm text-sm leading-6 text-white/70">{description}</p>
        </div>
        <div className="rounded-2xl border border-white/10 bg-black/20 p-3 text-white shadow-lg shadow-black/20">
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
  badge,
  tone = "default",
}: {
  title: string;
  description: string;
  href: string;
  icon: React.ReactNode;
  external?: boolean;
  badge: string;
  tone?: "default" | "violet" | "sky";
}) {
  const toneClasses =
    tone === "violet"
      ? "bg-[linear-gradient(135deg,rgba(200,89,144,0.2),rgba(255,255,255,0.02))]"
      : tone === "sky"
        ? "bg-[linear-gradient(135deg,rgba(88,160,255,0.18),rgba(255,255,255,0.02))]"
        : "bg-[linear-gradient(135deg,rgba(255,255,255,0.08),rgba(255,255,255,0.02))]";

  return (
    <Card className={`overflow-hidden border-white/10 bg-white/5 text-white shadow-2xl shadow-black/20 ${toneClasses}`}>
      <CardContent className="space-y-4 p-5">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-3">
            <div className="inline-flex rounded-2xl border border-white/10 bg-black/20 p-3 text-white">
              {icon}
            </div>
            <div className="space-y-2">
              <div className="inline-flex rounded-full border border-white/10 bg-white/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-white/70">
                {badge}
              </div>
              <p className="text-lg font-semibold">{title}</p>
              <p className="max-w-sm text-sm leading-6 text-white/70">{description}</p>
            </div>
          </div>
          <Button
            asChild
            variant="outline"
            className="shrink-0 rounded-full border-white/15 bg-white/10 text-white hover:bg-white/15"
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

        <div className="flex flex-wrap items-center gap-2 border-t border-white/10 pt-4 text-xs text-white/60">
          <span className="inline-flex rounded-full border border-white/10 bg-white/5 px-3 py-1">
            {badge}
          </span>
          <span className="inline-flex rounded-full border border-white/10 bg-white/5 px-3 py-1">
            {external ? "External" : "Internal"}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

function UserRolePill({ role }: { role: string }) {
  const isAdmin = role.toLowerCase() === "admin";
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ${
        isAdmin
          ? "bg-[#C85990]/20 text-[#FFD6E9]"
          : "bg-white/10 text-white/80"
      }`}
    >
      {role}
    </span>
  );
}

export default function AdminConsole() {
  const router = useRouter();
  const { user, isLoading, logout } = useAuth();
  const [users, setUsers] = useState<User[]>([]);
  const [loadingUsers, setLoadingUsers] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [learningReviews, setLearningReviews] = useState<SearchLearningReview[]>([]);
  const [learningAliases, setLearningAliases] = useState<SearchLearningAlias[]>([]);
  const [loadingLearning, setLoadingLearning] = useState(false);
  const [learningError, setLearningError] = useState<string | null>(null);

  const isAdmin = user?.role === "admin";
  const isCheckingSession = isLoading;

  useEffect(() => {
    const loadUsers = async () => {
      if (!isAdmin) return;

      const token = localStorage.getItem("token");
      if (!token) {
        setError("Admin session missing. Please sign in again.");
        return;
      }

      setLoadingUsers(true);
      setError(null);

      try {
        const response = await getAllUsers(token);
        setUsers(response.data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load users.");
      } finally {
        setLoadingUsers(false);
      }
    };

    loadUsers();
  }, [isAdmin]);

  useEffect(() => {
    const loadLearningQueue = async () => {
      if (!isAdmin) return;

      const token = localStorage.getItem("token");
      if (!token) {
        setLearningError("Admin session missing. Please sign in again.");
        return;
      }

      setLoadingLearning(true);
      setLearningError(null);

      try {
        const [reviewsResponse, aliasesResponse] = await Promise.all([
          getSearchLearningReviews(token, "pending", 12),
          getSearchLearningAliases(token, 20),
        ]);
        setLearningReviews(reviewsResponse.data);
        setLearningAliases(aliasesResponse.data);
      } catch (err) {
        setLearningError(err instanceof Error ? err.message : "Failed to load learning queue.");
      } finally {
        setLoadingLearning(false);
      }
    };

    loadLearningQueue();
  }, [isAdmin]);

  const summary = useMemo(
    () => ({
      users: users.length,
      admins: users.filter((entry) => entry.role === "admin").length,
      editors: users.filter((entry) => entry.role !== "admin").length,
    }),
    [users]
  );

  const learningSummary = useMemo(
    () => ({
      pendingReviews: learningReviews.length,
      learnedAliases: learningAliases.length,
      avgConfidence:
        learningReviews.length > 0
          ? Math.round(
              (learningReviews.reduce((total, review) => total + review.confidence, 0) /
                learningReviews.length) *
                100
            )
          : 0,
    }),
    [learningAliases.length, learningReviews]
  );

  const handleLogout = async () => {
    await logout();
    router.push("/");
  };

  const refreshLearningQueue = async () => {
    const token = localStorage.getItem("token");
    if (!token) return;

    const [reviewsResponse, aliasesResponse] = await Promise.all([
      getSearchLearningReviews(token, "pending", 12),
      getSearchLearningAliases(token, 20),
    ]);
    setLearningReviews(reviewsResponse.data);
    setLearningAliases(aliasesResponse.data);
  };

  const handleReviewAction = async (reviewId: number, action: "approve" | "reject") => {
    const token = localStorage.getItem("token");
    if (!token) {
      setLearningError("Admin session missing. Please sign in again.");
      return;
    }

    setLoadingLearning(true);
    setLearningError(null);

    try {
      if (action === "approve") {
        await approveSearchLearningReview(token, reviewId);
        toast.success("Review approved and saved to the learning queue");
      } else {
        await rejectSearchLearningReview(token, reviewId);
        toast.success("Review rejected");
      }
      await refreshLearningQueue();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to update learning review.";
      setLearningError(message);
      toast.error(message);
    } finally {
      setLoadingLearning(false);
    }
  };

  if (isCheckingSession) {
    return (
      <div className="min-h-svh bg-[radial-gradient(circle_at_top,_rgba(200,89,144,0.25),_transparent_28%),linear-gradient(180deg,#14070f_0%,#090509_100%)] px-6 py-16 text-white">
        <div className="mx-auto flex max-w-6xl items-center justify-center">
          <Card className="w-full max-w-xl border-white/10 bg-white/5 text-white shadow-2xl shadow-black/30">
            <CardContent className="p-8 text-center">
              <div className="mx-auto mb-4 h-12 w-12 animate-spin rounded-full border-2 border-white/20 border-t-white/80" />
              <p className="text-lg font-semibold">Checking administrator access</p>
              <p className="mt-2 text-sm text-white/70">
                Verifying your session before loading the platform console.
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
                Platform Admin
              </p>
              <h1 className="max-w-3xl font-serif text-4xl leading-tight text-white md:text-5xl">
                Sign in to manage the full Cost Savvy Health platform.
              </h1>
              <p className="max-w-2xl text-base leading-7 text-white/70">
                Administrators can review users, update site content, manage the
                healthcare data model, and keep the public experience current.
              </p>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <StatCard
                title="User access"
                value="Admin only"
                description="Protected sign-in for platform management."
                icon={<Users className="h-5 w-5" />}
              />
              <StatCard
                title="Content tools"
                value="Sanity"
                description="Edit home pages, blog posts, and editorial content."
                icon={<FileText className="h-5 w-5" />}
              />
            </div>

          <Card className="border-white/10 bg-white/5 text-white shadow-2xl shadow-black/20">
            <CardContent className="p-6">
                <div className="flex flex-wrap items-center gap-3 text-sm text-white/70">
                  <ShieldCheck className="h-5 w-5 text-[#FFD6E9]" />
                  <span>
                    Use your administrator account to unlock users, content, data,
                    and correspondence tools.
                  </span>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="lg:pt-10">
            <SignInForm
              authType={null}
              onSwitch={() => undefined}
              redirectTo="/admin"
              hideSwitch
            />
          </div>
        </div>
      </div>
    );
  }

  if (!isAdmin) {
    return (
      <div className="min-h-svh bg-[radial-gradient(circle_at_top,_rgba(200,89,144,0.25),_transparent_28%),linear-gradient(180deg,#14070f_0%,#090509_100%)] px-6 py-16 text-white">
        <div className="mx-auto flex max-w-3xl items-center justify-center">
          <Card className="w-full border-white/10 bg-white/5 text-white shadow-2xl shadow-black/30">
            <CardHeader>
              <CardTitle className="text-2xl">Administrator access required</CardTitle>
              <CardDescription className="text-white/70">
                The account currently signed in does not have admin privileges.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-white/70">
                Please sign out and log back in with an administrator account to
                manage users, platform content, CPT codes, and hospital data.
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
                  <Link href="/">
                    Back to site
                  </Link>
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
              Admin Console
            </p>
            <h1 className="font-serif text-4xl leading-tight md:text-5xl">
              Welcome back, {user?.name || "administrator"}.
            </h1>
            <p className="max-w-2xl text-base leading-7 text-white/70">
              Manage users, editorial content, pricing data, and customer
              correspondence from one control center.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <Button asChild variant="outline" className="rounded-full border-white/15 bg-white/10 text-white hover:bg-white/15">
              <Link href={SANITY_STUDIO_URL} target="_blank" rel="noreferrer">
                Open Studio
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
            title="Users"
            value={loadingUsers ? "..." : String(summary.users)}
            description="Registered accounts in the platform."
            icon={<Users className="h-5 w-5" />}
            tone="violet"
          />
          <StatCard
            title="Admins"
            value={loadingUsers ? "..." : String(summary.admins)}
            description="Accounts with elevated access."
            icon={<ShieldCheck className="h-5 w-5" />}
            tone="emerald"
          />
          <StatCard
            title="Content"
            value="Sanity"
            description="Edit homepage, blog, and page content."
            icon={<FileText className="h-5 w-5" />}
            tone="sky"
          />
          <StatCard
            title="Data"
            value="Postgres"
            description="Healthcare search and rate tables live here."
            icon={<Database className="h-5 w-5" />}
            tone="default"
          />
        </div>

        <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <Card id="users" className="border-white/10 bg-white/5 text-white shadow-2xl shadow-black/20">
            <CardHeader className="border-b border-white/10">
              <CardTitle className="text-2xl">Users</CardTitle>
              <CardDescription className="text-white/70">
                Review the current user base and confirm administrator access.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 p-6">
              {error && (
                <div className="rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
                  {error}
                </div>
              )}

              {loadingUsers ? (
                <div className="rounded-2xl border border-white/10 bg-white/5 p-6 text-sm text-white/70">
                  Loading users...
                </div>
              ) : (
                <div className="overflow-hidden rounded-2xl border border-white/10">
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-white/10 text-left text-sm">
                      <thead className="bg-white/5 text-white/60">
                        <tr>
                          <th className="px-4 py-3 font-medium">Name</th>
                          <th className="px-4 py-3 font-medium">Email</th>
                          <th className="px-4 py-3 font-medium">Organization</th>
                          <th className="px-4 py-3 font-medium">ZIP</th>
                          <th className="px-4 py-3 font-medium">Role</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/10 bg-[#120a10]">
                        {users.length === 0 ? (
                          <tr>
                            <td className="px-4 py-4 text-white/70" colSpan={5}>
                              No users found yet.
                            </td>
                          </tr>
                        ) : (
                          users.slice(0, 12).map((entry) => (
                            <tr key={entry.id} className="hover:bg-white/5">
                              <td className="px-4 py-3 font-medium">
                                <div>{entry.name}</div>
                                <div className="mt-1 text-xs text-white/50">
                                  {entry.jobTitle || "No job title provided"}
                                </div>
                              </td>
                              <td className="px-4 py-3 text-white/70">
                                <div>{entry.email}</div>
                                <div className="mt-1 max-w-[16rem] truncate text-xs text-white/50">
                                  {entry.useCase || "No use case provided"}
                                </div>
                              </td>
                              <td className="px-4 py-3 text-white/70">
                                <div>{entry.companyName || "—"}</div>
                                <div className="mt-1 text-xs text-white/50">
                                  {entry.organizationType || "No organization type"}
                                </div>
                              </td>
                              <td className="px-4 py-3 text-white/70">
                                {entry.zipCode || "—"}
                              </td>
                              <td className="px-4 py-3">
                                <UserRolePill role={entry.role} />
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <div className="space-y-6">
            <ActionCard
              title="Content management"
              description="Open Sanity Studio to update homepage blocks, articles, and other editorial pages."
              href={SANITY_STUDIO_URL}
              icon={<FileText className="h-5 w-5" />}
              external
              badge="Sanity studio"
              tone="violet"
            />
            <ActionCard
              title="CPT and hospital data"
              description="Review the public search and pricing flows that read from the Postgres-backed healthcare tables."
              href="/quote"
              icon={<Hospital className="h-5 w-5" />}
              badge="Data tools"
              tone="sky"
            />
            <ActionCard
              title="Correspondence"
              description="Open the support mailbox and contact flow. A persisted inbox can be added here next."
              href="mailto:Chat@costsavvy.health"
              icon={<MessageSquare className="h-5 w-5" />}
              external
              badge="Inbox"
              tone="default"
            />
          </div>
        </div>

        <div className="grid gap-6 xl:grid-cols-2">
          <Card id="learning-queue" className="overflow-hidden border-white/10 bg-white/5 text-white shadow-2xl shadow-black/20">
            <div className="border-b border-white/10 bg-[linear-gradient(135deg,rgba(200,89,144,0.2),rgba(255,255,255,0.02))] px-6 py-5">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="space-y-2">
                  <div className="inline-flex items-center rounded-full border border-white/10 bg-white/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.24em] text-white/70">
                    Learning queue
                  </div>
                  <CardTitle className="text-2xl">Review suggestions before they become live aliases</CardTitle>
                  <CardDescription className="max-w-2xl text-white/70">
                    Low-confidence search learnings are routed here so you can approve only the good matches and keep the data set clean.
                  </CardDescription>
                </div>
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="rounded-2xl border border-white/10 bg-black/20 px-4 py-3">
                    <div className="text-xs uppercase tracking-[0.18em] text-white/50">Pending</div>
                    <div className="mt-1 text-2xl font-semibold">{learningSummary.pendingReviews}</div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-black/20 px-4 py-3">
                    <div className="text-xs uppercase tracking-[0.18em] text-white/50">Aliases</div>
                    <div className="mt-1 text-2xl font-semibold">{learningSummary.learnedAliases}</div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-black/20 px-4 py-3">
                    <div className="text-xs uppercase tracking-[0.18em] text-white/50">Avg</div>
                    <div className="mt-1 text-2xl font-semibold">{learningSummary.avgConfidence}%</div>
                  </div>
                </div>
              </div>
            </div>
            <CardContent className="space-y-4 p-6">
              {learningError && (
                <div className="rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
                  {learningError}
                </div>
              )}

              {loadingLearning ? (
                <div className="rounded-2xl border border-white/10 bg-white/5 p-6 text-sm text-white/70">
                  Loading learning queue...
                </div>
              ) : learningReviews.length === 0 ? (
                <div className="rounded-3xl border border-dashed border-white/15 bg-white/5 px-5 py-8 text-sm text-white/70">
                  <div className="flex items-center gap-3">
                    <div className="rounded-full border border-white/10 bg-white/10 p-2">
                      <CheckCircle2 className="h-4 w-4 text-emerald-300" />
                    </div>
                    <div>
                      <p className="font-medium text-white">Nothing waiting for review</p>
                      <p className="mt-1 text-white/60">
                        Low-confidence suggestions will appear here when the learning model is unsure.
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  {learningReviews.map((review) => {
                    const suggestion =
                      review.suggestedAlias ||
                      review.suggestedHospitalName ||
                      "Needs review";
                    const confidence = Math.round(review.confidence * 100);
                    const confidenceTone =
                      confidence >= 80
                        ? "bg-emerald-500/15 text-emerald-200 border-emerald-400/20"
                        : confidence >= 60
                          ? "bg-amber-500/15 text-amber-200 border-amber-400/20"
                          : "bg-rose-500/15 text-rose-200 border-rose-400/20";

                    return (
                      <div
                        key={review.id}
                        className="rounded-3xl border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.02))] p-5 shadow-lg shadow-black/20"
                      >
                        <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                          <div className="space-y-4">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="inline-flex rounded-full border border-white/10 bg-white/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-white/70">
                                {review.source}
                              </span>
                              <span className={`inline-flex rounded-full border px-3 py-1 text-xs font-semibold ${confidenceTone}`}>
                                {confidence}% confidence
                              </span>
                              <span className="inline-flex rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-white/70">
                                {review.resultCount} result{review.resultCount === 1 ? "" : "s"}
                              </span>
                            </div>

                            <div>
                              <p className="text-base font-semibold leading-6 text-white">
                                {review.queryText}
                              </p>
                              <p className="mt-2 text-sm leading-6 text-white/60">
                                {review.rationale || "No rationale provided by the model."}
                              </p>
                            </div>

                            <div className="flex flex-wrap gap-2 text-xs">
                              {review.cptCode && (
                                <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-white/75">
                                  CPT {review.cptCode}
                                </span>
                              )}
                              {review.zipCode && (
                                <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-white/75">
                                  ZIP {review.zipCode}
                                </span>
                              )}
                              {review.insurer && (
                                <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-white/75">
                                  {review.insurer}
                                </span>
                              )}
                            </div>
                          </div>

                          <div className="min-w-[20rem] space-y-3 rounded-3xl border border-white/10 bg-black/20 p-4">
                            <div className="text-xs uppercase tracking-[0.2em] text-white/50">Suggested learning</div>
                            <div className="space-y-2">
                              <div className="rounded-2xl border border-white/10 bg-white/5 px-3 py-3">
                                <div className="text-[11px] uppercase tracking-[0.18em] text-white/50">
                                  Alias
                                </div>
                                <div className="mt-1 text-sm font-medium text-white">
                                  {review.suggestedAlias || "Not provided"}
                                </div>
                              </div>
                              <div className="rounded-2xl border border-white/10 bg-white/5 px-3 py-3">
                                <div className="text-[11px] uppercase tracking-[0.18em] text-white/50">
                                  Hospital
                                </div>
                                <div className="mt-1 text-sm font-medium text-white">
                                  {review.suggestedHospitalName || review.hospitalName || "Not provided"}
                                </div>
                              </div>
                            </div>

                            <div className="flex flex-wrap gap-2 pt-1">
                              <Button
                                type="button"
                                size="sm"
                                className="rounded-full bg-emerald-500/90 px-4 text-white hover:bg-emerald-500"
                                onClick={() => handleReviewAction(review.id, "approve")}
                              >
                                <CheckCircle2 className="h-4 w-4" />
                                Approve
                              </Button>
                              <Button
                                type="button"
                                size="sm"
                                variant="outline"
                                className="rounded-full border-white/15 bg-white/10 px-4 text-white hover:bg-white/15"
                                onClick={() => handleReviewAction(review.id, "reject")}
                              >
                                <XCircle className="h-4 w-4" />
                                Reject
                              </Button>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>

          <Card id="learned-aliases" className="overflow-hidden border-white/10 bg-white/5 text-white shadow-2xl shadow-black/20">
            <div className="border-b border-white/10 bg-[linear-gradient(135deg,rgba(88,160,255,0.18),rgba(255,255,255,0.02))] px-6 py-5">
              <div className="space-y-2">
                <div className="inline-flex items-center rounded-full border border-white/10 bg-white/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.24em] text-white/70">
                  Learned aliases
                </div>
                <CardTitle className="text-2xl">Approved matches that now improve search results</CardTitle>
                <CardDescription className="max-w-2xl text-white/70">
                  These are the aliases that have been approved and are now available to the live search path.
                </CardDescription>
              </div>
            </div>
            <CardContent className="space-y-4 p-6">
              {loadingLearning ? (
                <div className="rounded-2xl border border-white/10 bg-white/5 p-6 text-sm text-white/70">
                  Loading learned aliases...
                </div>
              ) : learningAliases.length === 0 ? (
                <div className="rounded-3xl border border-dashed border-white/15 bg-white/5 px-5 py-8 text-sm text-white/70">
                  <div className="flex items-center gap-3">
                    <div className="rounded-full border border-white/10 bg-white/10 p-2">
                      <FileText className="h-4 w-4 text-sky-300" />
                    </div>
                    <div>
                      <p className="font-medium text-white">No aliases approved yet</p>
                      <p className="mt-1 text-white/60">
                        Once you approve a review, it will appear here and immediately help future searches.
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  {learningAliases.map((alias) => {
                    const target =
                      alias.aliasType === "service"
                        ? `${alias.codeType || "CPT"} ${alias.code || ""}`.trim()
                        : alias.hospitalName || "Hospital";
                    return (
                      <div
                        key={`${alias.aliasType}-${alias.id}`}
                        className="flex flex-col gap-4 rounded-3xl border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.02))] p-4 md:flex-row md:items-center md:justify-between"
                      >
                        <div className="space-y-2">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="inline-flex rounded-full border border-white/10 bg-white/10 px-3 py-1 text-xs font-semibold text-white/80">
                              {alias.aliasType}
                            </span>
                            <span className="inline-flex rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-white/60">
                              {target}
                            </span>
                          </div>
                          <div className="text-base font-medium text-white">{alias.aliasText}</div>
                          <div className="text-sm text-white/55">{alias.sourceQuery}</div>
                        </div>
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="inline-flex rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-semibold text-white/75">
                            {(alias.confidence * 100).toFixed(0)}%
                          </span>
                          <span className="inline-flex rounded-full border border-emerald-400/20 bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-200">
                            Live
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
