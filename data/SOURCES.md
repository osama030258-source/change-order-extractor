# Sample document sources

All samples are excerpts of public-record documents published by local governments. Nothing here is private or client data.

| sample file | original document | what the excerpt is |
|---|---|---|
| `samples/glenwood_2025_cos.txt` | https://cityofglenwood.org/wp-content/uploads/2024/12/05.15.2025-Special-Meeting-Packet.pdf (Glenwood, Iowa city council packet, 15 May 2025) | Text of CO #5R and CO #09 only |
| `samples/everett_2009_co3.txt` | https://lfportal.everettwa.gov/WebLink/0/doc/684252/Page23.aspx (Everett, WA council packet, 18 Feb 2009) | One OCR'd page of Change Order No. 3 |
| `samples/anchorage_2005_excerpt.txt` | https://apps.asdk12.org/School_Board/archives/Arc2004-2005/Agendas/20050509/J02M273.pdf (Anchorage School District memo #273, 9 May 2005) | Three of eight change orders: the summary-table rows, then their detail pages (order preserved) |

## Important caveats

- These `.txt` files are the **text layer as returned by a web-fetch tool on 2026-10-17**, not the original PDF bytes. They are good for a first run, but for the real submission download the original PDFs, put them in `data/samples/` and re-check the labels against them. The pipeline reads PDFs directly.
- Labels in `data/labels/` were written by reading that text. A human should verify them against the originals before quoting any accuracy number.
- Check each source site's terms before redistributing documents publicly; if unsure, keep only the URL and the labels in the repo and ask users to download the PDFs.
