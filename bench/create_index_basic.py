import lancedb

# 주석 해제하고 올바른 주소 넣을 것 db = lancedb.connect("")
# 주석 해제하고 올바른 주소 넣을 것 tbl = db.open_table("")

print("기존 인덱스:", tbl.list_indices())

# 새 방식: vector_column만 지정
tbl.create_index(
    vector_column="embedding",   # 벡터 컬럼 이름
    num_partitions=256,
    num_sub_vectors=64
)

print("새 인덱스:", tbl.list_indices())
