import { AuthProvider } from "@/context/AuthContext";
import NextTopLoader from "nextjs-toploader";

export default function WithoutNavbarLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <>
      <NextTopLoader color="#FFFFFF" height={3} showSpinner={false} />
      <AuthProvider>{children}</AuthProvider>
    </>
  );
}
