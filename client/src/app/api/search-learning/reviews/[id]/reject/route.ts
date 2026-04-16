import { NextResponse } from "next/server";
import { query } from "@/lib/postgres";
import { requireAdmin } from "@/lib/admin-auth";
import { ensureSearchLearningSchema } from "@/lib/search-learning";

export const runtime = "nodejs";

export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const admin = await requireAdmin(request);
  if (!admin) {
    return NextResponse.json({ success: false, message: "Forbidden" }, { status: 403 });
  }

  await ensureSearchLearningSchema();
  const { id } = await params;
  const reviewId = Number(id);
  if (!Number.isFinite(reviewId)) {
    return NextResponse.json({ success: false, message: "Invalid review id" }, { status: 400 });
  }

  const current = await query<{ status: string }>(
    `SELECT status FROM search_learning_reviews WHERE id = $1 LIMIT 1`,
    [reviewId]
  );
  const review = current.rows[0];
  if (!review) {
    return NextResponse.json({ success: false, message: "Review not found" }, { status: 404 });
  }
  if (review.status !== "pending") {
    return NextResponse.json(
      { success: false, message: `Review already ${review.status}` },
      { status: 400 }
    );
  }

  await query(
    `
      UPDATE search_learning_reviews
      SET status = 'rejected', reviewed_at = now()
      WHERE id = $1
    `,
    [reviewId]
  );

  return NextResponse.json({
    success: true,
    message: "Review rejected.",
  });
}
