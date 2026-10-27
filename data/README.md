# BBC News corpus

The project uses the BBC News text corpus hosted by the UCD Machine Learning Group: http://mlg.ucd.ie/datasets/bbc.html . If the page is temporarily unavailable, retain a local copy of the original corpus from the project archive.

Place the original text files under these five directories:

| Directory | Topic |
|---|---|
| `data/bbc/business/` | Business |
| `data/bbc/entertainment/` | Entertainment |
| `data/bbc/politics/` | Politics |
| `data/bbc/sport/` | Sport |
| `data/bbc/tech/` | Technology |

Expected input: 2,225 text files. The current loader removes identical article texts before splitting, leaving 2,127 documents in the supplied snapshot. Articles are not redistributed in this repository; consult the source’s terms. Source authors: D. Greene and P. Cunningham, *Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering*, ICML 2006.

The checked-in run manifest records test-document hashes and class counts without publishing the article text. Near-duplicate stories can remain even after exact deduplication.
