import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

type BackendContext = {
  params: Promise<{
    path?: string[];
  }>;
};

const BACKEND_URL = process.env.API_URL ?? "http://127.0.0.1:8000";
const API_KEY = process.env.API_KEY ?? "";

async function proxy(request: NextRequest, context: BackendContext) {
  const { path = [] } = await context.params;
  const target = new URL(path.join("/"), `${BACKEND_URL.replace(/\/$/, "")}/`);
  target.search = request.nextUrl.search;

  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) {
    headers.set("content-type", contentType);
  }
  if (API_KEY) {
    headers.set("x-api-key", API_KEY);
  }
  const requestId = request.headers.get("x-request-id");
  if (requestId) {
    headers.set("x-request-id", requestId);
  }

  const hasBody = !["GET", "HEAD"].includes(request.method);
  const upstream = await fetch(target, {
    method: request.method,
    headers,
    body: hasBody ? await request.text() : undefined,
    cache: "no-store",
  });

  const responseHeaders = new Headers();
  const upstreamType = upstream.headers.get("content-type");
  const upstreamRequestId = upstream.headers.get("x-request-id");
  if (upstreamType) {
    responseHeaders.set("content-type", upstreamType);
  }
  if (upstreamRequestId) {
    responseHeaders.set("x-request-id", upstreamRequestId);
  }

  return new NextResponse(await upstream.text(), {
    status: upstream.status,
    headers: responseHeaders,
  });
}

export async function GET(request: NextRequest, context: BackendContext) {
  return proxy(request, context);
}

export async function POST(request: NextRequest, context: BackendContext) {
  return proxy(request, context);
}

export async function PUT(request: NextRequest, context: BackendContext) {
  return proxy(request, context);
}

export async function PATCH(request: NextRequest, context: BackendContext) {
  return proxy(request, context);
}

export async function DELETE(request: NextRequest, context: BackendContext) {
  return proxy(request, context);
}
