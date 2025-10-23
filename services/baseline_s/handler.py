import os, json, time, hashlib, base64, math
import boto3, numpy as np
from typing import Any, Dict, List, Tuple
from botocore.config import Config
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

# ─────────────────────────────────────────────────────────────────────────────
# 환경 변수
# ─────────────────────────────────────────────────────────────────────────────
REGION      = os.environ["AWS_REGION"]
S3_BUCKET   = os.environ["S3_BUCKET"]
LANCE_URI   = os.environ["LANCE_URI"]
LANCE_TABLE = os.environ.get("LANCE_TABLE", "chunks_dedup")
MODEL_EMBED = os.environ["MODEL_EMBED"]   # amazon.titan-embed-text-v2:0
MODEL_GEN   = os.environ["MODEL_GEN"]     # anthropic.claude-3-5-sonnet-20240620-v1:0
EMBED_DIM   = int(os.environ.get("EMBED_DIM", "512"))
TOP_K_DEF   = int(os.environ.get("TOP_K", "5"))
DDB_TABLE   = os.environ.get("DDB_TABLE")

# ─────────────────────────────────────────────────────────────────────────────
# AWS 클라이언트
# ─────────────────────────────────────────────────────────────────────────────
br  = boto3.client("bedrock-runtime", region_name=REGION, config=Config(retries={"max_attempts": 5}))
ssm = boto3.client("ssm", region_name=REGION)
ddb = boto3.resource("dynamodb", region_name=REGION).Table(DDB_TABLE) if DDB_TABLE else None

# LanceDB lazy load
_lancedb = None
_tbl = None

# OpenSearch (Serverless)
session = boto3.Session()
credentials = session.get_credentials()
awsauth = AWSV4SignerAuth(credentials, REGION, "aoss")

_os_client = None
_os_index = None

# SSM parameter cache
_SSM_CACHE: Dict[str,str] = {}
_SSM_TS = 0

# 컨테이너 전역 상태(웜/콜드 판별용)
_WARMED = False

# ─────────────────────────────────────────────────────────────────────────────
# SSM 유틸
# ─────────────────────────────────────────────────────────────────────────────
def ssm_get(name: str, optional: bool=False, default: str="") -> str:
    """
    SSM 파라미터 조회. 60초 캐시.
    """
    global _SSM_CACHE, _SSM_TS
    now = time.time()
    if now - _SSM_TS > 60:
        _SSM_CACHE = {}
        _SSM_TS = now
    if name in _SSM_CACHE:
        return _SSM_CACHE[name]
    try:
        v = ssm.get_parameter(Name=name)["Parameter"]["Value"]
    except Exception:
        if optional:
            return default
        raise
    _SSM_CACHE[name] = v
    return v

# ─────────────────────────────────────────────────────────────────────────────
# 비용 계산 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _to_float(s: str, default: float=0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default

def estimate_tokens(text: str) -> int:
    """
    토큰 추정(대략적인 휴리스틱): 문자수/4
    스트리밍/Bedrock에서 토큰 리턴이 없는 경우 근사치로 사용.
    """
    if not text:
        return 0
    # 한글/영문 섞임 고려해 4.0으로 보수적 근사
    return max(1, int(len(text) / 4.0))

def calc_lambda_cost_gbs(duration_ms: float, memory_mb: int, price_gb_s: float) -> Tuple[float, float]:
    """
    Lambda GB-s 및 비용 계산.
    - duration_ms: 요청 처리 소요(ms)
    - memory_mb : 메모리(MB)
    - price_gb_s: GB-s 당 가격(USD)
    """
    duration_s = max(0.0, duration_ms / 1000.0)
    gb_s = duration_s * (memory_mb / 1024.0)
    return gb_s, gb_s * price_gb_s

def build_cost_breakdown(
    *,
    lambda_duration_ms: float,
    lambda_memory_mb: int,
    apigw_requests: int,
    retrieve_provider: str,
    num_retrievals: int,
    bedrock_in_tokens: int,
    bedrock_out_tokens: int,
) -> Dict[str, Any]:
    """
    요청 1건에 대한 비용 추정치를 산출. (모든 단가는 SSM에서 가져오되, 없으면 0)
    """
    price_lambda_gb_s  = _to_float(ssm_get("/srag/PRICE_LAMBDA_GB_S",  optional=True, default="0"))
    price_apigw_req    = _to_float(ssm_get("/srag/PRICE_APIGW_PER_REQ", optional=True, default="0"))
    price_brin_1k      = _to_float(ssm_get("/srag/PRICE_BEDROCK_IN_PER_1K",  optional=True, default="0"))
    price_brout_1k     = _to_float(ssm_get("/srag/PRICE_BEDROCK_OUT_PER_1K", optional=True, default="0"))
    price_os_query     = _to_float(ssm_get("/srag/PRICE_OPENSEARCH_PER_QUERY", optional=True, default="0"))
    price_lance_query  = _to_float(ssm_get("/srag/PRICE_LANCEDB_PER_QUERY",   optional=True, default="0"))

    # Lambda
    lambda_gb_s, lambda_cost = calc_lambda_cost_gbs(lambda_duration_ms, lambda_memory_mb, price_lambda_gb_s)

    # API GW
    apigw_cost = apigw_requests * price_apigw_req

    # Retrieval
    if retrieve_provider == "os":
        retrieval_cost = num_retrievals * price_os_query
    else:
        retrieval_cost = num_retrievals * price_lance_query

    # Bedrock (토큰 단가: 1K 당)
    bedrock_in_cost  = (bedrock_in_tokens  / 1000.0) * price_brin_1k
    bedrock_out_cost = (bedrock_out_tokens / 1000.0) * price_brout_1k
    bedrock_cost     = bedrock_in_cost + bedrock_out_cost

    total_cost = lambda_cost + apigw_cost + retrieval_cost + bedrock_cost

    return {
        "lambda": {
            "duration_ms": lambda_duration_ms,
            "memory_mb": lambda_memory_mb,
            "gb_s": lambda_gb_s,
            "cost": round(lambda_cost, 8),
        },
        "apigw": {
            "requests": apigw_requests,
            "cost": round(apigw_cost, 8),
        },
        "retrieval": {
            "provider": retrieve_provider,
            "queries": num_retrievals,
            "cost": round(retrieval_cost, 8),
        },
        "bedrock": {
            "input_tokens": bedrock_in_tokens,
            "output_tokens": bedrock_out_tokens,
            "cost_input": round(bedrock_in_cost, 8),
            "cost_output": round(bedrock_out_cost, 8),
            "cost_total": round(bedrock_cost, 8),
        },
        "total_cost": round(total_cost, 8),
    }

# ─────────────────────────────────────────────────────────────────────────────
# CloudWatch Embedded Metric Format (EMF)
# ─────────────────────────────────────────────────────────────────────────────
def emit_emf(namespace: str, metrics: Dict[str, float], dimensions: Dict[str, str]):
    """
    CloudWatch EMF 포맷으로 메트릭을 로깅.
    """
    emf = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": namespace,
                "Dimensions": [list(dimensions.keys())],
                "Metrics": [{"Name": k, "Unit": "None"} for k in metrics.keys()]
            }]
        }
    }
    emf.update(dimensions)
    emf.update(metrics)
    print(json.dumps(emf, ensure_ascii=False))

# ─────────────────────────────────────────────────────────────────────────────
# 초기화 (LanceDB / OpenSearch)
# ─────────────────────────────────────────────────────────────────────────────
def init_lancedb():
    global _lancedb, _tbl
    if _lancedb is None:
        import lancedb
        _lancedb = lancedb.connect(LANCE_URI)
        _tbl = _lancedb.open_table(LANCE_TABLE)

def ensure_lance_hnsw_index():
    """
    LanceDB 테이블에 IVF_HNSW_PQ 인덱스 보장
    """
    init_lancedb()
    try:
        idxes = _tbl.list_indices()
        print("DEBUG LanceDB indices (before):", idxes)
        has_hnsw = any(i.get("type") in ("HNSW","IVF_HNSW_PQ","IVF_HNSW_SQ") for i in idxes)
        if not has_hnsw:
            print("Building IVF_HNSW_PQ index on LanceDB table...")
            _tbl.create_index(
                column="vector",
                index_type="IVF_HNSW_PQ",
                metric="cosine",
                num_partitions=256,
                num_sub_vectors=32,
                replace=True
            )
            print("DEBUG LanceDB indices (after):", _tbl.list_indices())
    except Exception as e:
        print("WARN: HNSW index creation failed", e)

def init_opensearch():
    global _os_client, _os_index
    if _os_client is None:
        host = ssm_get("/srag/OS_HOST", optional=True)
        if not host:
            raise RuntimeError("OS_HOST not set")
        _os_index = ssm_get("/srag/OS_INDEX", optional=True) or "rag-vec-512"
        _os_client = OpenSearch(
            hosts=[{"host": host.replace("https://", "").replace("http://", ""), "port": 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )

# ─────────────────────────────────────────────────────────────────────────────
# 임베딩
# ─────────────────────────────────────────────────────────────────────────────
def titan_embed(text: str) -> np.ndarray:
    body = {"inputText": text, "dimensions": EMBED_DIM, "normalize": True}
    for a in range(3):
        try:
            r = br.invoke_model(
                modelId=MODEL_EMBED,
                body=json.dumps(body).encode("utf-8"),
                accept="application/json",
                contentType="application/json",
            )
            j = json.loads(r["body"].read())
            return np.array(j["embedding"], dtype="float32")
        except Exception:
            if a == 2: raise
            time.sleep(1.0 * (a + 1))

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval (Lance / OpenSearch)
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_by_vec(qvec: np.ndarray, k: int) -> Tuple[List[Dict[str, Any]], float, str]:
    """
    반환: (passages, search_duration_sec, provider)
    """
    provider = ssm_get("/srag/RETRIEVAL_PROVIDER", optional=True) or "lance"
    if provider == "os":
        init_opensearch()
        body = {
            "size": k,
            "query": {"knn": {"vec": {"vector": qvec.tolist(), "k": k}}},
            "_source": ["id", "text", "source", "meta"],
        }
        start = time.time()
        resp = _os_client.search(index=_os_index, body=body)
        end = time.time()
        dur = end - start
        print(f"DEBUG search_duration: {dur:.3f} sec (OpenSearch)")
        hits = resp["hits"]["hits"]
        passages = [{
            "id": h["_id"],
            "text": h["_source"].get("text", ""),
            "source": h["_source"].get("source"),
            "meta": h["_source"].get("meta"),
            "score": float(h.get("_score", 0.0)),
        } for h in hits]
        return passages, dur, "os"
    else:
        init_lancedb()
        ensure_lance_hnsw_index()
        start = time.time()
        rows = (
            _tbl.search(qvec)
                .metric("cosine")
                .select(["id", "text", "source", "meta"])
                .limit(k)
                .to_list()
        )
        end = time.time()
        dur = end - start
        print(f"DEBUG search_duration: {dur:.3f} sec (LanceDB)")
        passages = [{
            "id": r.get("id"),
            "text": r.get("text"),
            "source": r.get("source"),
            "meta": r.get("meta"),
            "score": float(r.get("_distance") or 0.0),
        } for r in rows]
        return passages, dur, "lance"

# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def compress_text(t: str, max_sent: int = 3, max_chars: int = 360) -> str:
    t = t.replace("\n", " ").strip()
    sents = [s.strip() for s in t.split(". ") if s.strip()]
    sents = sents[:max_sent]
    out = ". ".join(sents)
    if len(out) > max_chars:
        out = out[:max_chars] + "…"
    return out

def qkey(query: str = "", qvec: np.ndarray = None, top_k: int = 5) -> str:
    h = hashlib.sha256()
    h.update(str(top_k).encode())
    if query: h.update(query.encode("utf-8"))
    if qvec is not None:
        h.update(b"|vec|")
        h.update(np.array(qvec, dtype="float32").tobytes())
    return h.hexdigest()

def ddb_get(key: str):
    if not ddb: return None
    r = ddb.get_item(Key={"qhash": key})
    if "Item" not in r: return None
    try:
        return json.loads(r["Item"]["result"])
    except Exception:
        return None

def ddb_put(key: str, value: Any, ttl_min: int):
    if not ddb: return
    ttl = int(time.time()) + ttl_min * 60
    ddb.put_item(Item={"qhash": key, "result": json.dumps(value, ensure_ascii=False), "ttl": ttl})

# ─────────────────────────────────────────────────────────────────────────────
# LLM 호출 (TTFB/토큰 추정 로깅)
# ─────────────────────────────────────────────────────────────────────────────
def call_llm(prompt: str, max_tokens: int = 200) -> Tuple[str, float, int, int]:
    """
    반환: (full_text, ttfb_sec, in_tokens_est, out_tokens_est)
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    # 입력 토큰 근사 (프롬프트 기준)
    in_tokens_est = estimate_tokens(prompt)

    # 스트리밍 호출 (TTFB 측정)
    start = time.time()
    stream = br.invoke_model_with_response_stream(
        modelId=MODEL_GEN,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )
    first_token_time = None
    full_text = ""

    for event in stream["body"]:
        if "chunk" in event:
            chunk = json.loads(event["chunk"]["bytes"])
            if first_token_time is None:
                first_token_time = time.time()
                print(f"DEBUG TTFB: {first_token_time - start:.3f} sec")
            # 베드록 스트리밍 형식에서 텍스트 추출
            part = ""
            if "content" in chunk and isinstance(chunk["content"], list) and chunk["content"]:
                part = chunk["content"][0].get("text", "")
            elif "delta" in chunk and isinstance(chunk["delta"], dict):
                part = chunk["delta"].get("text", "")
            full_text += part or ""

    ttfb = (first_token_time - start) if first_token_time else 0.0
    out_tokens_est = estimate_tokens(full_text)
    return full_text, ttfb, in_tokens_est, out_tokens_est

def _json_ok(obj: Any) -> Dict[str, Any]:
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json; charset=utf-8"},
        "body": json.dumps(obj, ensure_ascii=False)
    }

# ─────────────────────────────────────────────────────────────────────────────
# Lambda Handler
# ─────────────────────────────────────────────────────────────────────────────
def main(event, context):
    global _WARMED
    cold = 0 if _WARMED else 1
    if not _WARMED:
        _WARMED = True  # 첫 호출 이후부터는 웜으로 간주

    if event.get("warmup"):
        return {"statusCode": 200, "body": "warm"}

    req_start = time.time()
    path = (event.get("path") or "").lower()
    method = (event.get("httpMethod") or "post").upper()

    print("DEBUG event['body']:", event.get("body"))

    # 공통 SSM 설정
    use_cache  = (ssm_get("/srag/USE_DDB_CACHE", optional=True) == "true")
    ttl_min    = int(ssm_get("/srag/CACHE_TTL_MIN", optional=True) or "10")
    use_comp   = (ssm_get("/srag/USE_CTX_COMPRESS", optional=True) == "true")

    # ========================= /retrieve =========================
    if path.endswith("/retrieve") and method in ("GET", "POST"):
        if method == "GET":
            qs = event.get
            qs = event.get("queryStringParameters") or {}
            query = (qs.get("q") or "").strip()
            top_k = int(qs.get("top_k") or TOP_K_DEF)
            qvec = None
        else:
            body = json.loads(event.get("body") or "{}")
            query = (body.get("query") or "").strip()
            top_k = int(body.get("top_k") or TOP_K_DEF)
            qvec = body.get("qvec")

        if not query and not qvec:
            return {"statusCode": 400, "body": json.dumps({"error": "query or qvec required"})}

        # 임베딩 생성 or 벡터 직접 사용
        embed_start = time.time()
        if qvec is None:
            qvec_np = titan_embed(query)
            qvec = qvec_np.tolist()
        else:
            qvec_np = np.array(qvec, dtype="float32")
            qvec = qvec_np.tolist()
        embed_end = time.time()

        key = qkey(query=query, qvec=np.array(qvec, dtype="float32"), top_k=top_k)
        if use_cache:
            hit = ddb_get(key)
            if hit is not None:
                resp = {"passages": hit, "cache": True}
                emit_emf(
                    "SRAG/Metrics",
                    metrics={"Requests": 1, "CacheHit": 1, "Cold": cold},
                    dimensions={"Path": "/retrieve"}
                )
                return _json_ok(resp)

        print("DEBUG qvec length:", len(qvec))
        print("DEBUG raw qvec:", qvec[:10])

        # 검색
        search_start = time.time()
        passages, search_dur, provider = retrieve_by_vec(np.array(qvec, dtype="float32"), top_k)
        search_end = time.time()
        print(f"DEBUG total_search_duration: {search_end - search_start:.3f} sec")

        if use_comp:
            for p in passages:
                p["text"] = compress_text(p.get("text",""))

        if use_cache:
            ddb_put(key, passages, ttl_min)

        # 요청 종료 및 비용/지표 산출
        req_end = time.time()
        lambda_duration_ms = (req_end - req_start) * 1000.0
        memory_mb = context.memory_limit_in_mb if hasattr(context, "memory_limit_in_mb") else 0

        # 비용 산출 (Bedrock 미호출)
        cost = build_cost_breakdown(
            lambda_duration_ms=lambda_duration_ms,
            lambda_memory_mb=int(memory_mb or 0),
            apigw_requests=1,
            retrieve_provider=provider,
            num_retrievals=1,
            bedrock_in_tokens=0,
            bedrock_out_tokens=0,
        )

        emit_emf(
            "SRAG/Metrics",
            metrics={
                "Requests": 1,
                "Cold": cold,
                "EmbedSec": max(0.0, embed_end - embed_start),
                "SearchSec": max(0.0, search_dur),
                "LatencySec": max(0.0, req_end - req_start),
                "LambdaCostUSD": cost["lambda"]["cost"],
                "TotalCostUSD": cost["total_cost"],
            },
            dimensions={"Path": "/retrieve", "Provider": provider}
        )

        return _json_ok({
            "passages": passages,
            "metrics": {
                "cold": cold,
                "embed_sec": round(embed_end - embed_start, 4),
                "search_sec": round(search_dur, 4),
                "latency_sec": round(req_end - req_start, 4),
                "provider": provider,
            },
            "cost": cost
        })

    # ========================= /query =========================
    body = json.loads(event.get("body") or "{}")
    query = (body.get("query") or "").strip()
    top_k = int(body.get("top_k") or TOP_K_DEF)
    max_tokens = int(body.get("max_tokens") or 80)

    if not query:
        return {"statusCode": 400, "body": json.dumps({"error": "query required"})}

    # 임베딩
    embed_start = time.time()
    qvec = titan_embed(query)
    embed_end = time.time()

    print("DEBUG qvec length:", len(qvec))
    print("DEBUG raw qvec:", qvec[:10])

    # 검색
    search_start = time.time()
    passages, search_dur, provider = retrieve_by_vec(qvec, top_k)
    search_end = time.time()
    print(f"DEBUG total_search_duration: {search_end - search_start:.3f} sec")

    # 컨텍스트 구성
    use_comp = (ssm_get("/srag/USE_CTX_COMPRESS", optional=True) == "true")
    ctx_txt = "\n\n".join(
        f"[{i+1}] {compress_text(p['text']) if use_comp else p['text']}"
        for i, p in enumerate(passages)
    )
    prompt = (
        "Answer the question concisely in English based only on the following context.\n\n"
        f"Context:\n{ctx_txt}\n\n"
        f"Question: {query}\nAnswer:"
    )

    # LLM 호출
    gen_start = time.time()
    answer, ttfb_sec, in_tok, out_tok = call_llm(prompt, max_tokens=max_tokens)
    gen_end = time.time()

    # 요청 종료 및 비용/지표 산출
    req_end = time.time()
    lambda_duration_ms = (req_end - req_start) * 1000.0
    memory_mb = context.memory_limit_in_mb if hasattr(context, "memory_limit_in_mb") else 0

    cost = build_cost_breakdown(
        lambda_duration_ms=lambda_duration_ms,
        lambda_memory_mb=int(memory_mb or 0),
        apigw_requests=1,
        retrieve_provider=provider,
        num_retrievals=1,
        bedrock_in_tokens=in_tok,
        bedrock_out_tokens=out_tok,
    )

    emit_emf(
        "SRAG/Metrics",
        metrics={
            "Requests": 1,
            "Cold": cold,
            "EmbedSec": max(0.0, embed_end - embed_start),
            "SearchSec": max(0.0, search_dur),
            "GenSec": max(0.0, gen_end - gen_start),
            "TTFB": max(0.0, ttfb_sec),
            "LatencySec": max(0.0, req_end - req_start),
            "InTokens": in_tok,
            "OutTokens": out_tok,
            "LambdaCostUSD": cost["lambda"]["cost"],
            "BedrockCostUSD": cost["bedrock"]["cost_total"],
            "TotalCostUSD": cost["total_cost"],
        },
        dimensions={"Path": "/query", "Provider": provider}
    )

    return _json_ok({
        "answer": answer,
        "passages": passages,
        "metrics": {
            "cold": cold,
            "embed_sec": round(embed_end - embed_start, 4),
            "search_sec": round(search_dur, 4),
            "gen_sec": round(gen_end - gen_start, 4),
            "ttfb_sec": round(ttfb_sec, 4),
            "latency_sec": round(req_end - req_start, 4),
            "in_tokens_est": in_tok,
            "out_tokens_est": out_tok,
            "provider": provider,
        },
        "cost": cost
    })