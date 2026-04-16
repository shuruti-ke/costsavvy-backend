// Define types
export interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  avatar: string | null;
  phoneNumber?: string | null;
  companyName?: string | null;
  jobTitle?: string | null;
  organizationType?: string | null;
  zipCode?: string | null;
  useCase?: string | null;
  organizationSize?: string | null;
  specialty?: string | null;
  primaryGoal?: string | null;
  accountType?: string | null;
}

export type IntakeType = "consumer" | "employer" | "provider" | "payer";

export interface RegisterUserInput {
  name: string;
  email: string;
  password: string;
  phoneNumber?: string;
  companyName?: string;
  jobTitle?: string;
  organizationType?: string;
  zipCode?: string;
  useCase?: string;
  organizationSize?: string;
  specialty?: string;
  primaryGoal?: string;
  accountType?: "business" | "consumer";
}

export interface RegisterResult {
  success: boolean;
  message: string;
  confirmationRequired?: boolean;
  dashboardPath?: string;
}

export interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<{ token: string; user: User }>;
  register: (input: RegisterUserInput) => Promise<RegisterResult>;
  logout: () => Promise<void>;
  error: string | null;
}
