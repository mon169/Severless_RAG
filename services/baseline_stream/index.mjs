import { BedrockRuntimeClient, InvokeModelWithResponseStreamCommand } from "@aws-sdk/client-bedrock-runtime";
import { fetch } from "undici";
import { StandardRetryStrategy } from "@smithy/middleware-retry";

const REGION            = process.env.AWS_REGION || "ap-northeast-2";
const MODEL_GEN         = process.env.MODEL_GEN;
const RETRIEVE_URL      = process.env.RETRIEVE_URL;
const STREAM_TOPK       = parseInt(process.env.STREAM_TOPK || "5", 10);
const STREAM_MAX_TOKENS = parseInt(process.env.STREAM_MAX_TOKENS || "80", 10);
if (!MODEL_GEN) throw new Error("MODEL_GEN not set");
if (!RETRIEVE_URL) throw new Error("RETRIEVE_URL not set");

const br = new BedrockRuntimeClient({ region: REGION, retryStrategy: new StandardRetryStrategy(async () => 5) });

async function fetchPassages(q, topk) {
  const url = `${RETRIEVE_URL}?q=${encodeURIComponent(q)}&top_k=${topk}`;
  const r = await fetch(url, { method: "GET" });
  if (!r.ok) throw new Error(`retrieve failed: HTTP ${r.status}`);
  const j = await r.json();
  return j.passages || [];
}

export const handler = async (event) => {
  try {
    const q = (event.queryStringParameters?.q || "").trim();
    const k = parseInt(event.queryStringParameters?.k || `${STREAM_TOPK}`, 10);
    if (!q) return { statusCode: 400, body: "q required" };

    const passages = await fetchPassages(q, k);
    const ctx = passages.map((p, i) => `[${i + 1}] ${p.text}`).join("\n\n");

    const prompt =
`Please answer concisely in English.

contexts:
${ctx}

Q : ${q}
A :`;

    const body = JSON.stringify({
      anthropic_version: "bedrock-2023-05-31",
      messages: [{ role: "user", content: [{ type: "text", text: prompt }] }],
      max_tokens: STREAM_MAX_TOKENS,
      temperature: 0.2
    });

    const cmd = new InvokeModelWithResponseStreamCommand({
      modelId: MODEL_GEN,
      contentType: "application/json",
      accept: "application/json",
      body
    });

    const res = await br.send(cmd);

    const chunks = [];
    for await (const evt of res.body) {
      if (evt.chunk) chunks.push(Buffer.from(evt.chunk.bytes).toString("utf-8"));
    }
    return { statusCode: 200, headers: { "Content-Type": "text/plain; charset=utf-8" }, body: chunks.join("") };
  } catch (err) {
    console.error("Stream error:", err);
    const code = err.$metadata?.httpStatusCode || 500;
    return { statusCode: code, headers: { "Content-Type": "application/json" }, body: JSON.stringify({ error: err.name || "Error", message: err.message || String(err) }) };
  }
};
