# Durable Model Archives

LoRA Manager can preserve a user-owned copy of model-site metadata beside each model
so a model remains understandable after its source page or account disappears.
This archive is separate from both LoRA Manager's editable `.metadata.json`
sidecar and the optional shared CivitAI metadata archive database.

Enable **Settings → Library → Metadata → Archive downloaded model metadata**.
Each archive stays beside its model so copying or backing up the model directory
also preserves its usage instructions and media.

## Layout

Each archive directory uses the model's file stem plus `.archive`:

```text
models/loras/
  example.safetensors
  example.metadata.json
  example.preview.webp
  example.archive/
    manifest.json
    description.html
    snapshots/
      <UTC timestamp>-<content digest>.json
    media/
      preview.webp
      example-...
      remote-...
```

- Snapshot JSON files are append-only. Identical source metadata is
  content-deduplicated instead of producing another snapshot.
- Library scans skip `.archive` directories, so saved media cannot be mistaken
  for models and does not slow normal model discovery.
- `description.html` is a convenient current rendering of the model and version
  descriptions. Snapshot JSON remains the canonical history.
- `manifest.json` records source IDs and URLs, snapshot history, local media,
  successful downloads, and failed downloads.
- Failed remote media remains in the manifest and is retried by a later model
  download or metadata fetch.
- Full and incremental cache rebuilds repair snapshots and copy media that is
  already available locally. Metadata fetch also retries remote example media.

The archive never writes, modifies, or deletes third-party `.civitai.info`
files.
