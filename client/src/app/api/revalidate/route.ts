// src/app/api/revalidate/route.ts

import { NextResponse } from "next/server"
import { revalidatePath } from "next/cache"

export async function POST(request: Request) {
  // grab the header named "secret"
  const incomingSecret = request.headers.get("secret")

  if (incomingSecret !== process.env.SANITY_REVALIDATE_SECRET) {
    return NextResponse.json(
      { message: "Invalid token" },
      { status: 401 }
    )
  }
  await revalidatePath("/blog")
  await revalidatePath("/blog/article/[slug]", "page")
  await revalidatePath("/blog/mainCard/[slug]", "page")
  await revalidatePath("/blog/other/[slug]", "page")

  return NextResponse.json({ revalidated: true })
}
