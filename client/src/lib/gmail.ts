import nodemailer, { type Transporter } from "nodemailer";

type NewUserNotification = {
  name: string;
  email: string;
  role: string;
  userId: string;
};

type SignupConfirmation = {
  name: string;
  email: string;
  confirmationUrl: string;
  dashboardPath?: string;
  accountType: "business" | "consumer";
  organizationType?: string | null;
};

type BusinessIntakeNotification = {
  contactName: string;
  email: string;
  companyName: string;
  organizationType: string;
  uploadType: string;
  notes: string;
  fileName?: string | null;
  attachment?: {
    filename: string;
    content: Buffer;
    contentType?: string;
  } | null;
};

let transporter: Transporter | null = null;

function getRequiredEnv(name: string) {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing ${name} environment variable`);
  }
  return value;
}

function getTransporter() {
  if (!transporter) {
    transporter = nodemailer.createTransport({
      host: process.env.GMAIL_SMTP_HOST || "smtp.gmail.com",
      port: Number(process.env.GMAIL_SMTP_PORT || 465),
      secure: (process.env.GMAIL_SMTP_PORT || "465") === "465",
      auth: {
        user: getRequiredEnv("GMAIL_USER"),
        pass: getRequiredEnv("GMAIL_APP_PASSWORD"),
      },
    });
  }

  return transporter;
}

export async function sendNewUserNotificationEmail({
  name,
  email,
  role,
  userId,
}: NewUserNotification) {
  const fromName = process.env.GMAIL_FROM_NAME || "Cost Savvy Health";
  const fromEmail = getRequiredEnv("GMAIL_USER");
  const toEmail = process.env.NEW_USER_NOTIFICATION_TO || fromEmail;

  const subject = `New user signup: ${name}`;
  const text = [
    "A new user registered on Cost Savvy Health.",
    "",
    `Name: ${name}`,
    `Email: ${email}`,
    `Role: ${role}`,
    `User ID: ${userId}`,
  ].join("\n");

  const html = `
    <div style="font-family: Arial, sans-serif; line-height: 1.5; color: #1f2937;">
      <h2 style="margin: 0 0 16px;">New user signup</h2>
      <p style="margin: 0 0 16px;">A new user registered on Cost Savvy Health.</p>
      <table cellpadding="0" cellspacing="0" style="border-collapse: collapse;">
        <tr><td style="padding: 4px 12px 4px 0; font-weight: bold;">Name</td><td style="padding: 4px 0;">${name}</td></tr>
        <tr><td style="padding: 4px 12px 4px 0; font-weight: bold;">Email</td><td style="padding: 4px 0;">${email}</td></tr>
        <tr><td style="padding: 4px 12px 4px 0; font-weight: bold;">Role</td><td style="padding: 4px 0;">${role}</td></tr>
        <tr><td style="padding: 4px 12px 4px 0; font-weight: bold;">User ID</td><td style="padding: 4px 0;">${userId}</td></tr>
      </table>
    </div>
  `;

  await getTransporter().sendMail({
    from: `${fromName} <${fromEmail}>`,
    to: toEmail,
    subject,
    text,
    html,
  });
}

export async function sendSignupConfirmationEmail({
  name,
  email,
  confirmationUrl,
  dashboardPath,
  accountType,
  organizationType,
}: SignupConfirmation) {
  const fromName = process.env.GMAIL_FROM_NAME || "Cost Savvy Health";
  const fromEmail = getRequiredEnv("GMAIL_USER");
  const subject = `Confirm your Cost Savvy Health account`;
  const dashboardHint = dashboardPath
    ? `After confirming, you can sign in at ${dashboardPath} to reach your dashboard.`
    : "After confirming, sign in to access your account.";
  const accountLabel =
    accountType === "business"
      ? `Business account${organizationType ? ` (${organizationType})` : ""}`
      : "Consumer account";

  const text = [
    `Hi ${name},`,
    "",
    "Thanks for creating your Cost Savvy Health account.",
    `Account type: ${accountLabel}`,
    "",
    `Confirm your email here: ${confirmationUrl}`,
    "",
    dashboardHint,
    "",
    "If you did not request this account, you can ignore this email.",
  ].join("\n");

  const html = `
    <div style="font-family: Arial, sans-serif; line-height: 1.6; color: #1f2937;">
      <h2 style="margin: 0 0 16px;">Confirm your Cost Savvy Health account</h2>
      <p style="margin: 0 0 12px;">Hi ${name},</p>
      <p style="margin: 0 0 12px;">Thanks for creating your Cost Savvy Health account.</p>
      <p style="margin: 0 0 12px;"><strong>Account type:</strong> ${accountLabel}</p>
      <p style="margin: 0 0 20px;">Click the button below to confirm your email address and activate your account.</p>
      <p style="margin: 0 0 24px;">
        <a href="${confirmationUrl}" style="display:inline-block;background:#8C2F5D;color:#ffffff;text-decoration:none;padding:12px 20px;border-radius:999px;font-weight:bold;">Confirm email</a>
      </p>
      <p style="margin: 0 0 12px;">${dashboardHint}</p>
      <p style="margin: 0;">If you did not request this account, you can ignore this email.</p>
    </div>
  `;

  await getTransporter().sendMail({
    from: `${fromName} <${fromEmail}>`,
    to: email,
    subject,
    text,
    html,
  });
}

export async function sendBusinessIntakeEmail({
  contactName,
  email,
  companyName,
  organizationType,
  uploadType,
  notes,
  fileName,
  attachment,
}: BusinessIntakeNotification) {
  const fromName = process.env.GMAIL_FROM_NAME || "Cost Savvy Health";
  const fromEmail = getRequiredEnv("GMAIL_USER");
  const toEmail = process.env.NEW_USER_NOTIFICATION_TO || fromEmail;
  const subject = `Business data intake request: ${companyName}`;
  const text = [
    "A business account submitted a data intake request.",
    "",
    `Contact: ${contactName}`,
    `Email: ${email}`,
    `Company: ${companyName}`,
    `Organization type: ${organizationType}`,
    `Upload type: ${uploadType}`,
    fileName ? `File: ${fileName}` : "File: none provided",
    "",
    "Notes:",
    notes || "No additional notes provided.",
  ].join("\n");

  const html = `
    <div style="font-family: Arial, sans-serif; line-height: 1.6; color: #1f2937;">
      <h2 style="margin: 0 0 16px;">Business data intake request</h2>
      <table cellpadding="0" cellspacing="0" style="border-collapse: collapse;">
        <tr><td style="padding: 4px 12px 4px 0; font-weight: bold;">Contact</td><td style="padding: 4px 0;">${contactName}</td></tr>
        <tr><td style="padding: 4px 12px 4px 0; font-weight: bold;">Email</td><td style="padding: 4px 0;">${email}</td></tr>
        <tr><td style="padding: 4px 12px 4px 0; font-weight: bold;">Company</td><td style="padding: 4px 0;">${companyName}</td></tr>
        <tr><td style="padding: 4px 12px 4px 0; font-weight: bold;">Organization type</td><td style="padding: 4px 0;">${organizationType}</td></tr>
        <tr><td style="padding: 4px 12px 4px 0; font-weight: bold;">Upload type</td><td style="padding: 4px 0;">${uploadType}</td></tr>
        <tr><td style="padding: 4px 12px 4px 0; font-weight: bold;">File</td><td style="padding: 4px 0;">${fileName || "None provided"}</td></tr>
      </table>
      <p style="margin: 16px 0 8px; font-weight: bold;">Notes</p>
      <p style="margin: 0;">${(notes || "No additional notes provided.").replace(/\n/g, "<br />")}</p>
    </div>
  `;

  await getTransporter().sendMail({
    from: `${fromName} <${fromEmail}>`,
    to: toEmail,
    subject,
    text,
    html,
    attachments: attachment
      ? [
          {
            filename: attachment.filename,
            content: attachment.content,
            contentType: attachment.contentType,
          },
        ]
      : undefined,
  });
}
