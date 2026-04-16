"use client";

import React, { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowUpRight,
  Database,
  FileText,
  Hospital,
  LogOut,
  MessageSquare,
  ShieldCheck,
  Users,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import SignInForm from "@/components/auth/sign-in-form";
import { useAuth } from "@/context/AuthContext";
import { getAllUsers } from "@/api/auth/api";
import type { User } from "@/types/context/auth-user";

const SANITY_STUDIO_URL = "https://cost-savy.sanity.studio/structure";

function StatCard({
  title,
  value,
  description,
  icon,
}: {
  title: string;
  value: string;
  description: string;
  icon: React.ReactNode;
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
  icon: React.ReactNode;
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

  const summary = useMemo(
    () => ({
      users: users.length,
      admins: users.filter((entry) => entry.role === "admin").length,
      editors: users.filter((entry) => entry.role !== "admin").length,
    }),
    [users]
  );

  const handleLogout = async () => {
    await logout();
    router.push("/");
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
          />
          <StatCard
            title="Admins"
            value={loadingUsers ? "..." : String(summary.admins)}
            description="Accounts with elevated access."
            icon={<ShieldCheck className="h-5 w-5" />}
          />
          <StatCard
            title="Content"
            value="Sanity"
            description="Edit homepage, blog, and page content."
            icon={<FileText className="h-5 w-5" />}
          />
          <StatCard
            title="Data"
            value="Postgres"
            description="Healthcare search and rate tables live here."
            icon={<Database className="h-5 w-5" />}
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
            />
            <ActionCard
              title="CPT and hospital data"
              description="Review the public search and pricing flows that read from the Postgres-backed healthcare tables."
              href="/quote"
              icon={<Hospital className="h-5 w-5" />}
            />
            <ActionCard
              title="Correspondence"
              description="Open the support mailbox and contact flow. A persisted inbox can be added here next."
              href="mailto:Chat@costsavvy.health"
              icon={<MessageSquare className="h-5 w-5" />}
              external
            />
          </div>
        </div>
      </div>
    </div>
  );
}
