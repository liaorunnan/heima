from rdflib import Graph

# 1. 创建一个图对象
g = Graph()

print("正在加载数据，请稍候...")

# 2. 解析文件
# format="turtle" 指定格式，如果是 .xml 或 .nt 格式需相应修改
# 如果文件很大（几百MB以上），这一步可能会比较慢，且占用大量内存
g.parse("./data/firstreleasetriple.ttl", format="turtle")

print(f"加载完成！图中共有 {len(g)} 个三元组。")

# 3. 简单遍历：打印前 5 条数据看看长什么样
# s=Subject(主语), p=Predicate(谓语), o=Object(宾语)
count = 0
for s, p, o in g:
    print(f"主语: {s}")
    print(f"谓语: {p}")
    print(f"宾语: {o}")
    print("-" * 30)

    count += 1
    if count >= 5:
        break