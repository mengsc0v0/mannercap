# 当前数据划分与泄漏检查

日期：2026-09-19  
状态：development-only；正式四类数据用途尚未冻结。

已为当前 64 条 D0 源轨迹和 seed42 LA4VLA 微拟合的 64 个派生窗口建立统一
split manifest，共 128 条记录：

| 类型 | 数量 | split |
| --- | ---: | --- |
| D0 source episode | 64 | development |
| LA4VLA derived window | 64 | development |
| 合计 | 128 | development |

自动检查把 `pair_id`、完整 lineage、初态、scene/对象/物理配置和语言模板族作为
不可拆分组；派生窗口必须引用清单内的源 episode，并继承其 split 和全部分组字段。
本次检查覆盖 72 个分组，跨 split 泄漏为 0。

工件：`reports/gate2/current_split_manifest_20260919.json`。内部 canonical manifest
SHA-256 为 `d251a1001c754495574ac90cabc4c0fb9e63b7bb0469f5ad91a51a4d787d425e`；
格式化 JSON 文件 SHA-256 为
`beac1c8af0e664362e5e4f94c2095972e95b9963fa43c8d4c277b6d7c496406d`。

清单中 generator test、student train、student validation 和 student test 的数量都为
0。这是正确的当前状态：现有 D0 及其所有派生窗口只作开发/校准，不能在以后改名为
封存测试。正式请求生成前，仍需为新请求预注册其用途，再切窗口；同 lineage、pair、
初态或模板族的记录不能跨用途。
