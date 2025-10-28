# 计算机科学领域名词处理

## 学术前沿领域名词

为了获取学术界研究的一些前沿领域关键词，我们从会议官网中收集了近 10 年的体系结构四大会议（HPCA、ASPLOS、MICRO、ISCA）的论文摘要。格式分为从 IEEE 网站中（HPCA）提取的 `.csv` 与 ACM Digital Library 中（其余三个会）提取的 `.bib`。

> 注意：MICRO'22 因为特殊原因只在 IEEE 收录，因此合并到 HPCA 中

批量加载这些摘要：

```bash
# Parse & load abstracts
python3 loader.py
```

之后我们简单对摘要里的词汇做了 n-gram 词频统计，可以按频率排序得到高频关键词（包括 1-gram 单词，2/3-gram 短语）。

> 注意：这部分使用 Vibe Coding，可能需要仔细查看

```bash
python3 ntlk.py
```

与现有的第三版名词进行比对防止重复：

```bash
python3 diff.py
```

## 相关书籍中的名词提取

我们从《Memory System: Cache, DRAM, Disk》中的附录使用 OCR + 后处理提取出了近千个存储领域相关名词，经过去重+筛选最终结果详见 [Memory_Systems_Cache_DRAM_Disk](Memory_Systems_Cache_DRAM_Disk/)。