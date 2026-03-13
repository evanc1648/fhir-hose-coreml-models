"""
Re-export the UDOP decoder to CoreML — Attempt #5: full-sequence [1,64] with fp16-safe masking.

Previous attempts:
  #1 [1,1] fixed shape — stateless, repeating tokens
  #2 RangeDim — Espresso "Data-dependent shapes were disabled"
  #3 [1,64] + decoder_attention_mask — NaN/broken cross-attn (float16 overflow in masks)
  #4 [1,1] + finfo patch — works! cosine sim 0.999994, but stateless (no history)

This attempt: combine #3's full-sequence approach with #4's finfo patch.
The [1,64] shape was never the problem — the NaN from -3.4e38 overflowing to -inf in
float16 was. With the finfo patch, the decoder_attention_mask approach should work
correctly, giving the model full self-attention history for autoregressive decoding.
"""

import os
import torch
import numpy as np
import coremltools as ct

MODEL_DIR = "model-hf"
OUTPUT_DIR = "output"
IMAGE_SIZE = 512
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 1024
ENC_OUT_LEN = 128 + NUM_PATCHES  # 1152
D_MODEL = 1024
VOCAB_SIZE = 33201
MAX_DEC_LEN = 64

# Float16-safe minimum value (instead of float32's -3.4e38 which overflows in fp16)
FLOAT16_MIN = -65504.0


def load_model():
    from transformers import UdopForConditionalGeneration
    print("Loading UDOP model...")
    model = UdopForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    model.eval()
    return model


class FullSeqDecoder(torch.nn.Module):
    """UDOP decoder with full-sequence [1,64] input and attention mask."""
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.lm_head = model.lm_head
        self.config = model.config

    def forward(self, decoder_input_ids, decoder_attention_mask,
                encoder_hidden_states, encoder_attention_mask):
        out = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=None,
            use_cache=False,
            return_dict=True,
        )
        sequence_output = out.last_hidden_state
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.config.d_model ** -0.5)
        logits = self.lm_head(sequence_output)
        return logits


def patch_finfo_for_float16_safe_masking():
    """Monkey-patch torch.finfo so that .min returns -65504 for float32."""
    _original_finfo = torch.finfo

    class FakeFinfo:
        def __init__(self, real_finfo):
            self._real = real_finfo

        def __getattr__(self, name):
            if name == 'min':
                return FLOAT16_MIN
            return getattr(self._real, name)

    def patched_finfo(dtype):
        real = _original_finfo(dtype)
        if dtype in (torch.float32, torch.float64):
            return FakeFinfo(real)
        return real

    torch.finfo = patched_finfo
    return _original_finfo


def restore_finfo(original):
    torch.finfo = original


def trace_decoder(model, dec_start, pad_id):
    """Trace the decoder with finfo patch active."""
    wrapper = FullSeqDecoder(model)
    wrapper.eval()

    trace_num_real = 4
    trace_dec_ids = torch.full((1, MAX_DEC_LEN), pad_id, dtype=torch.long)
    trace_dec_ids[0, :trace_num_real] = torch.tensor([dec_start, 3, 9, 3])
    trace_dec_mask = torch.zeros(1, MAX_DEC_LEN, dtype=torch.long)
    trace_dec_mask[0, :trace_num_real] = 1
    trace_enc_hidden = torch.randn(1, ENC_OUT_LEN, D_MODEL)
    trace_enc_mask = torch.ones(1, ENC_OUT_LEN, dtype=torch.long)

    orig_finfo = patch_finfo_for_float16_safe_masking()

    with torch.no_grad():
        test_logits = wrapper(trace_dec_ids, trace_dec_mask, trace_enc_hidden, trace_enc_mask)
        print(f"  Wrapper output shape: {test_logits.shape}")
        print(f"  Logits range: [{test_logits.min().item():.2f}, {test_logits.max().item():.2f}]")
        print(f"  Has NaN: {torch.isnan(test_logits).any().item()}")

        print("  Tracing...")
        traced = torch.jit.trace(
            wrapper, (trace_dec_ids, trace_dec_mask, trace_enc_hidden, trace_enc_mask),
        )
        traced_logits = traced(trace_dec_ids, trace_dec_mask, trace_enc_hidden, trace_enc_mask)
        print(f"  Traced vs eager max diff: {(test_logits - traced_logits).abs().max().item():.6f}")

    restore_finfo(orig_finfo)
    return traced, wrapper


def convert_and_verify(traced, wrapper, dec_start, pad_id, precision):
    """Convert traced model to CoreML with given precision and verify."""
    precision_label = "FLOAT16" if precision == ct.precision.FLOAT16 else "FLOAT32"
    print(f"\n=== Converting to CoreML ({precision_label}) ===")

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="decoder_input_ids", shape=(1, MAX_DEC_LEN), dtype=np.int32),
            ct.TensorType(name="decoder_attention_mask", shape=(1, MAX_DEC_LEN), dtype=np.int32),
            ct.TensorType(name="encoder_hidden_states", shape=(1, ENC_OUT_LEN, D_MODEL), dtype=np.float16),
            ct.TensorType(name="encoder_attention_mask", shape=(1, ENC_OUT_LEN), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float16),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=precision,
    )

    decoder_path = os.path.join(OUTPUT_DIR, f"UdopDecoder_{precision_label}.mlpackage")
    mlmodel.save(decoder_path)
    print(f"  Saved to: {decoder_path}")

    # Verify
    print(f"\n=== Verifying CoreML ({precision_label}) ===")
    loaded = ct.models.MLModel(decoder_path)

    # Use a fixed seed for reproducible comparison
    rng = np.random.RandomState(42)
    enc_hs_np = rng.randn(1, ENC_OUT_LEN, D_MODEL).astype(np.float16)
    enc_mask_np = np.ones((1, ENC_OUT_LEN), dtype=np.int32)

    num_real = 5
    dec_ids_np = np.full((1, MAX_DEC_LEN), pad_id, dtype=np.int32)
    dec_ids_np[0, :num_real] = [dec_start, 3, 9, 3, 19668]
    dec_mask_np = np.zeros((1, MAX_DEC_LEN), dtype=np.int32)
    dec_mask_np[0, :num_real] = 1

    result = loaded.predict({
        "decoder_input_ids": dec_ids_np,
        "decoder_attention_mask": dec_mask_np,
        "encoder_hidden_states": enc_hs_np,
        "encoder_attention_mask": enc_mask_np,
    })
    coreml_logits = result["logits"]
    has_nan = np.isnan(coreml_logits).any()
    has_inf = np.isinf(coreml_logits).any()
    print(f"  Shape: {coreml_logits.shape}, NaN: {has_nan}, Inf: {has_inf}")

    # PyTorch reference
    dec_ids_pt = torch.from_numpy(dec_ids_np).long()
    dec_mask_pt = torch.from_numpy(dec_mask_np).long()
    enc_hs_pt = torch.from_numpy(enc_hs_np.astype(np.float32))
    enc_mask_pt = torch.from_numpy(enc_mask_np).long()

    with torch.no_grad():
        pt_logits = wrapper(dec_ids_pt, dec_mask_pt, enc_hs_pt, enc_mask_pt).numpy()

    if not has_nan:
        pos = num_real - 1
        coreml_at_pos = coreml_logits[0, pos].astype(np.float32)
        pt_at_pos = pt_logits[0, pos]

        max_diff = np.abs(coreml_at_pos - pt_at_pos).max()
        cos_sim = np.dot(coreml_at_pos, pt_at_pos) / (
            np.linalg.norm(coreml_at_pos) * np.linalg.norm(pt_at_pos) + 1e-8
        )

        coreml_top5 = np.argsort(coreml_at_pos)[-5:][::-1]
        pt_top5 = np.argsort(pt_at_pos)[-5:][::-1]

        print(f"  Pos {pos}: max_diff={max_diff:.4f}, cosine={cos_sim:.6f}")
        print(f"  CoreML top 5: {coreml_top5.tolist()}")
        print(f"  PyTorch top 5: {pt_top5.tolist()}")
        print(f"  Top-1 match: {coreml_top5[0] == pt_top5[0]}")

        # Also check multiple positions
        for p in range(num_real):
            c = coreml_logits[0, p].astype(np.float32)
            r = pt_logits[0, p]
            cs = np.dot(c, r) / (np.linalg.norm(c) * np.linalg.norm(r) + 1e-8)
            md = np.abs(c - r).max()
            print(f"    pos {p}: cosine={cs:.6f}, max_diff={md:.4f}, "
                  f"CoreML top1={c.argmax()}, PyTorch top1={r.argmax()}")

        return cos_sim, max_diff, decoder_path, mlmodel
    else:
        print("  NaN in output!")
        return 0.0, float('inf'), decoder_path, mlmodel


def main():
    model = load_model()
    dec_start = model.config.decoder_start_token_id or model.config.pad_token_id
    pad_id = model.config.pad_token_id or 0
    print(f"  decoder_start_token_id: {dec_start}, pad_token_id: {pad_id}")

    # Trace
    print(f"\n=== Tracing Full-Sequence Decoder [1, {MAX_DEC_LEN}] ===")
    traced, wrapper = trace_decoder(model, dec_start, pad_id)

    # Test both precisions
    cos_f32, diff_f32, path_f32, ml_f32 = convert_and_verify(
        traced, wrapper, dec_start, pad_id, ct.precision.FLOAT32
    )
    cos_f16, diff_f16, path_f16, ml_f16 = convert_and_verify(
        traced, wrapper, dec_start, pad_id, ct.precision.FLOAT16
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  FLOAT32: cosine={cos_f32:.6f}, max_diff={diff_f32:.4f}")
    print(f"  FLOAT16: cosine={cos_f16:.6f}, max_diff={diff_f16:.4f}")

    # Pick the best one that meets criteria, preferring FLOAT16 for perf
    if cos_f16 >= 0.999:
        print("\n  -> Using FLOAT16 (meets cosine > 0.999)")
        chosen_path = path_f16
        chosen_model = ml_f16
    elif cos_f32 >= 0.999:
        print("\n  -> Using FLOAT32 (FLOAT16 precision too low, FLOAT32 meets criteria)")
        chosen_path = path_f32
        chosen_model = ml_f32
    else:
        print("\n  -> WARNING: neither precision meets cosine > 0.999")
        print("     Using FLOAT32 as it has higher cosine")
        chosen_path = path_f32
        chosen_model = ml_f32

    # Save as the final UdopDecoder.mlpackage
    final_path = os.path.join(OUTPUT_DIR, "UdopDecoder.mlpackage")
    chosen_model.author = "UDOP"
    chosen_model.short_description = (
        f"UDOP Decoder - full-sequence [{MAX_DEC_LEN}] with attention mask, "
        "fp16-safe masking, no KV cache"
    )
    chosen_model.save(final_path)
    print(f"\n  Final model saved to: {final_path}")

    print("\nSwift usage:")
    print("  1. Start with decoder_input_ids = [start_token, pad, pad, ..., pad] (len 64)")
    print("     and decoder_attention_mask = [1, 0, 0, ..., 0]")
    print("  2. Read logits at position 0, argmax -> next_token")
    print("  3. Set decoder_input_ids = [start_token, next_token, pad, ..., pad]")
    print("     and decoder_attention_mask = [1, 1, 0, ..., 0]")
    print("  4. Read logits at position 1, argmax -> next_token")
    print("  5. Repeat until EOS (token 1) or 64 tokens")


if __name__ == "__main__":
    main()
