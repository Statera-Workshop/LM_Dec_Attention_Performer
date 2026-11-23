# LM_Dec_Attention_Performer

Performer - Attention family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Runnable**
- Failure class: Architecture Unsuitable: not a causal language model (later tokens change earlier outputs, leak 6.3e-03)

## Provenance

- Source repository: (new) lucidrains/performer-pytorch
- Upstream / license: lucidrains/performer-pytorch (MIT)

## Differences from the source ([Unified] protocol)

- causal FAVOR+ linear attention with rotary positions

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 19.17 M
- Causality check: FAIL (largest change of earlier outputs 6.3e-03)
- Overfit check (4 x 256 bytes): 5.813 -> 0.001 nats, pass
- Peak GPU memory at length 2048 (batch 1): train 3955 MB, inference 122 MB
- Peak GPU memory at length 8192 (batch 1): train 15989 MB, inference 395 MB

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy performer-pytorch==1.1.4
python run.py check performer
python run.py lm02 performer      # enwik8 is downloaded to data/ on first use
```
