const CORS_HEADERS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, HEAD, OPTIONS",
  "access-control-allow-headers": "content-type",
};

const IMMUTABLE = "public, max-age=31536000, immutable";

function withHeaders(response, requestUrl) {
  const wrapped = new Response(response.body, response);
  for (const [key, value] of Object.entries(CORS_HEADERS)) {
    wrapped.headers.set(key, value);
  }
  if (!response.ok) {
    wrapped.headers.set("cache-control", "no-store");
    wrapped.headers.set("content-type", "text/plain; charset=utf-8");
    return wrapped;
  }
  if (response.ok && (requestUrl.pathname.endsWith(".net") ||
      requestUrl.pathname.endsWith(".js") ||
      requestUrl.pathname.endsWith(".png"))) {
    wrapped.headers.set("cache-control", IMMUTABLE);
    wrapped.headers.set(
      "content-type",
      requestUrl.pathname.endsWith(".js")
        ? "application/javascript; charset=utf-8"
        : requestUrl.pathname.endsWith(".png")
          ? "image/png"
          : "application/octet-stream",
    );
  } else if (requestUrl.pathname.endsWith(".json")) {
    wrapped.headers.set("cache-control", "public, max-age=300");
    wrapped.headers.set("content-type", "application/json; charset=utf-8");
  }
  return wrapped;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }
    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("method not allowed", {
        status: 405,
        headers: CORS_HEADERS,
      });
    }
    const response = await env.ASSETS.fetch(request);
    return withHeaders(response, url);
  },
};
