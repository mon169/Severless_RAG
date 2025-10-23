import lancedb
import time

# 주석 해제하고 올바른 주소 넣을 것 db = lancedb.connect("")
# 주석 해제하고 올바른 주소 넣을 것 tbl = db.open_table("")

print("현재 스키마:", tbl.schema)
print("총 행 수:", tbl.count_rows())
print("기존 인덱스:", tbl.list_indices())

def ensure_hnsw_index():
    idxes = tbl.list_indices()
    has_hnsw = any(i.index_type in ("IVF_HNSW_SQ", "IVF_HNSW_PQ") for i in idxes)

    if not has_hnsw:
        print("HNSW 계열 인덱스 없음 → 새로 생성 시도")

        tbl.create_index(
            metric="cosine",
            index_type="IVF_HNSW_SQ",
            vector_column_name="embedding"
        )

        index_name = "embedding_idx"

        print("인덱스 생성 진행 중")
        while True:
            stats = tbl.index_stats(index_name)
            done = stats.num_indexed_rows
            remain = stats.num_unindexed_rows
            total = done + remain
            print(f"   → 진행률 : {done:,} / {total:,} ({done/(total+1e-9):.2%})")

            if remain == 0:
                break
            time.sleep(5)

        print("인덱스 생성 완료:", tbl.list_indices())
    else:
        print("이미 HNSW 계열 인덱스 존재:", idxes)

if __name__ == "__main__":
    ensure_hnsw_index()
