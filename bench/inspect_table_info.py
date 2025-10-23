import lancedb

# 주석 해제 하고 올바른 주소 넣을 것 db = lancedb.connect("")

# 올바른 테이블 열기
tbl = db.open_table("chunks_dedup")

print("Schema:", tbl.schema)
print("Row count:", tbl.count_rows())
print(tbl.list_indices())