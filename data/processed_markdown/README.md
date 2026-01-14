# Processed Markdown Files

This folder contains Markdown files generated from the source PDFs.

## Output Format

Each processed PDF generates a Markdown file with:
- YAML frontmatter (source path, page count, table count)
- Clean, structured Markdown content
- Preserved tables in Markdown format
- Headers and sections maintained

## Files

Processed files will appear here after running:
```bash
python -m src.rag_pipeline --ingest
```
