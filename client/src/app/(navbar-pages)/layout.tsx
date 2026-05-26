import Navbar from "@/components/navbar";
import Footer from "@/components/footer";
import { AuthProvider } from "@/context/AuthContext";
import { Toaster } from "@/components/ui/sonner";
import NextTopLoader from "nextjs-toploader";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <>
      <NextTopLoader color="#FFFFFF" height={3} showSpinner={false} />
      <AuthProvider>
        <Navbar />
        {children}
        <Toaster />
        <Footer />
        <Toaster richColors />
      </AuthProvider>
    </>
  );
}
