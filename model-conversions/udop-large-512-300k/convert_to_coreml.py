"""
Convert UDOP (microsoft/udop-large-512-300k) to CoreML format for iOS deployment.

UDOP is a multimodal encoder-decoder (T5-based) model that takes:
  - text tokens + bounding boxes (from OCR)
  - document image (512x512)
and produces text output autoregressively.

We split into two CoreML models:
  1. Encoder: (input_ids, attention_mask, bbox, pixel_values) -> (hidden_states, encoder_attention_mask)
  2. Decoder: (decoder_input_ids, encoder_hidden_states, encoder_attention_mask) -> logits

The main challenge is that UDOP's `combine_image_text_embeddings` uses Python list
comprehensions and variable-length indexing that can't be traced. We rewrite it using
pure tensor operations: instead of filtering out "used" image patches, we keep all
patches and use attention masking to achieve the same effect.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

MODEL_DIR = "model-hf"
OUTPUT_DIR = "output"
IMAGE_SIZE = 512
PATCH_SIZE = 16
NUM_PATCHES_PER_DIM = IMAGE_SIZE // PATCH_SIZE  # 32
NUM_PATCHES = NUM_PATCHES_PER_DIM ** 2  # 1024


def load_model():
    from transformers import UdopForConditionalGeneration
    print("Loading UDOP model...")
    model = UdopForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    model.eval()
    return model


def make_visual_bbox():
    """Pre-compute the static visual bounding boxes for 512x512 / 16x16 patches.
    Returns tensor of shape [1024, 4] with values in [0, 1]."""
    steps = torch.linspace(0, 1, NUM_PATCHES_PER_DIM + 1)
    # x0, y0, x1, y1 for each patch in row-major order
    x0 = steps[:-1].repeat(NUM_PATCHES_PER_DIM, 1)  # [32, 32]
    y0 = steps[:-1].unsqueeze(1).repeat(1, NUM_PATCHES_PER_DIM)  # [32, 32]
    x1 = steps[1:].repeat(NUM_PATCHES_PER_DIM, 1)
    y1 = steps[1:].unsqueeze(1).repeat(1, NUM_PATCHES_PER_DIM)
    visual_bbox = torch.stack([x0, y0, x1, y1], dim=-1)  # [32, 32, 4]
    return visual_bbox.reshape(NUM_PATCHES, 4)  # [1024, 4]


def traceable_combine_image_text_embeddings(
    image_embeddings, inputs_embeds, bbox, attention_mask
):
    """Traceable replacement for combine_image_text_embeddings.

    Instead of filtering out image patches that overlap with OCR tokens (which
    requires dynamic indexing), we keep ALL patches and mask out the overlapping
    ones via attention_mask=0. The model sees the same effective representation.

    Args:
        image_embeddings: [B, 1024, d_model] - patch embeddings from CNN
        inputs_embeds: [B, seq_len, d_model] - text token embeddings
        bbox: [B, seq_len, 4] - bounding boxes for text tokens (0-1 range float)
        attention_mask: [B, seq_len] - text attention mask

    Returns:
        combined_embeds: [B, seq_len + 1024, d_model]
        combined_bbox: [B, seq_len + 1024, 4]
        combined_mask: [B, seq_len + 1024]
    """
    batch_size = inputs_embeds.shape[0]
    seq_len = inputs_embeds.shape[1]
    d_model = inputs_embeds.shape[2]

    # Step 1: Map each OCR token's bbox center to a patch index
    # bbox is in [0,1] range; map to patch grid coordinates
    center_x = ((bbox[:, :, 0] + bbox[:, :, 2]) / 2.0 * NUM_PATCHES_PER_DIM).long()
    center_x = center_x.clamp(0, NUM_PATCHES_PER_DIM - 1)
    center_y = ((bbox[:, :, 1] + bbox[:, :, 3]) / 2.0 * NUM_PATCHES_PER_DIM).long()
    center_y = center_y.clamp(0, NUM_PATCHES_PER_DIM - 1)
    ocr_patch_indices = center_y * NUM_PATCHES_PER_DIM + center_x  # [B, seq_len]

    # Step 2: Add vision embeddings at OCR token positions to text embeddings
    # Gather the corresponding patch embeddings for each text token
    gather_idx = ocr_patch_indices.unsqueeze(-1).expand(-1, -1, d_model)  # [B, seq_len, d_model]
    vision_at_text = torch.gather(image_embeddings, 1, gather_idx)  # [B, seq_len, d_model]

    # Zero out vision contribution for padding/special tokens (bbox all zeros or all ones)
    bbox_float = bbox.to(torch.float32)
    bbox_mean = bbox_float.mean(dim=-1)  # [B, seq_len]
    is_zero = (bbox_mean == 0.0).float()
    is_one = (bbox_mean == 1.0).float()
    is_special = (is_zero + is_one).clamp(max=1.0)  # 1.0 where special, 0.0 where real OCR
    is_real_ocr = 1.0 - is_special  # 1.0 where real OCR token
    vision_at_text = vision_at_text * is_real_ocr.unsqueeze(-1)

    inputs_embeds = inputs_embeds + vision_at_text

    # Step 3: Create patch attention mask - mask out patches that overlap with OCR tokens
    # Start with all patches visible
    patch_mask = torch.ones(batch_size, NUM_PATCHES, dtype=torch.float32,
                            device=attention_mask.device)
    # Mark patches that correspond to OCR tokens as masked (0)
    # Scatter: for each valid OCR token, set patch_mask at its patch index to 0
    patch_mask.scatter_(1, ocr_patch_indices, 1.0 - is_real_ocr)

    # Step 4: Concatenate text embeddings + ALL image patch embeddings
    combined_embeds = torch.cat([inputs_embeds, image_embeddings], dim=1)

    # Step 5: Concatenate bboxes - use pre-computed visual_bbox for patches
    visual_bbox = make_visual_bbox().to(bbox.device, dtype=bbox.dtype)
    visual_bbox = visual_bbox.unsqueeze(0).expand(batch_size, -1, -1)
    combined_bbox = torch.cat([bbox, visual_bbox], dim=1)

    # Step 6: Concatenate attention masks
    combined_mask = torch.cat([attention_mask.to(torch.float32), patch_mask], dim=1)

    return combined_embeds, combined_bbox, combined_mask


class TraceableEncoder(nn.Module):
    """Fully traceable UDOP encoder.

    Replaces the problematic combine_image_text_embeddings with a tensor-only version.
    """
    def __init__(self, model):
        super().__init__()
        encoder = model.encoder

        # Copy all submodules from the original encoder
        self.embed_tokens = encoder.embed_tokens
        self.embed_patches = encoder.embed_patches
        self.cell_2d_embedding = encoder.cell_2d_embedding
        self.relative_bias = encoder.relative_bias
        self.block = encoder.block
        self.final_layer_norm = encoder.final_layer_norm
        self.dropout = encoder.dropout
        self.config = encoder.config

    def forward(self, input_ids, attention_mask, bbox, pixel_values):
        # 1. Embed text tokens
        inputs_embeds = self.embed_tokens(input_ids)

        # 2. Embed image patches
        image_embeddings = self.embed_patches(pixel_values)

        # 3. Combine using our traceable function
        # bbox needs to be float for the combine function
        bbox_float = bbox.to(torch.float32) / 1000.0  # normalize from 0-1000 to 0-1 range
        hidden_states, combined_bbox, combined_mask = traceable_combine_image_text_embeddings(
            image_embeddings, inputs_embeds, bbox_float, attention_mask
        )

        # 4. Add 2D cell embeddings (only for encoder)
        hidden_states = hidden_states + self.cell_2d_embedding(combined_bbox)

        # 5. Compute relative position biases
        position_bias = self.relative_bias(attention_mask=combined_mask, bbox=combined_bbox)

        # 6. Create causal mask for encoder (non-causal, just padding mask)
        causal_mask = combined_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        causal_mask = (1.0 - causal_mask) * torch.finfo(hidden_states.dtype).min
        position_bias = position_bias + causal_mask

        # 7. Run through encoder layers
        hidden_states = self.dropout(hidden_states)

        for layer_module in self.block:
            layer_outputs = layer_module(
                hidden_states,
                causal_mask,
                position_bias,
                None,  # encoder_hidden_states
                None,  # encoder_extended_attention_mask
                None,  # encoder_decoder_position_bias
                None,  # layer_head_mask
                None,  # cross_attn_layer_head_mask
                None,  # past_key_value
                False,  # use_cache
                False,  # output_attentions
                None,   # cache_position
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states, combined_mask


class TraceableDecoder(nn.Module):
    """Traceable UDOP decoder + lm_head."""
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.lm_head = model.lm_head
        self.config = model.config

    def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask):
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
            use_cache=False,
        )
        sequence_output = decoder_outputs.last_hidden_state
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.config.d_model ** -0.5)
        logits = self.lm_head(sequence_output)
        return logits


def verify_encoder(model, traceable_encoder):
    """Verify the traceable encoder produces similar outputs to the original."""
    from transformers import UdopProcessor
    from PIL import Image

    print("\n=== Verifying Traceable Encoder ===")
    processor = UdopProcessor.from_pretrained(MODEL_DIR, apply_ocr=False)
    img = Image.new('RGB', (512, 512), color='white')
    words = ['Hello', 'world', 'test']
    boxes = [[100, 100, 200, 200], [300, 300, 400, 400], [500, 500, 600, 600]]
    encoding = processor(images=img, text="Question answering.", text_pair=words, boxes=boxes, return_tensors='pt')

    with torch.no_grad():
        # Original encoder
        orig_out = model.encoder(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            bbox=encoding['bbox'],
            pixel_values=encoding['pixel_values'],
            return_dict=True,
        )

        # Traceable encoder
        trace_hidden, trace_mask = traceable_encoder(
            encoding['input_ids'],
            encoding['attention_mask'],
            encoding['bbox'],
            encoding['pixel_values'],
        )

    # Compare shapes
    print(f"  Original shape: {orig_out.last_hidden_state.shape}, mask: {orig_out.attention_mask.shape}")
    print(f"  Traceable shape: {trace_hidden.shape}, mask: {trace_mask.shape}")

    # Note: outputs won't be identical because we keep all patches instead of filtering,
    # but the text token representations should be similar
    text_len = encoding['input_ids'].shape[1]
    orig_text = orig_out.last_hidden_state[:, :text_len, :]
    trace_text = trace_hidden[:, :text_len, :]
    cos_sim = torch.nn.functional.cosine_similarity(
        orig_text.flatten(), trace_text.flatten(), dim=0
    )
    print(f"  Text token cosine similarity: {cos_sim.item():.4f}")
    print(f"  (Values close to 1.0 indicate equivalent representations)")


def export_to_coreml(model):
    """Export encoder and decoder to CoreML via tracing."""
    import coremltools as ct

    traceable_encoder = TraceableEncoder(model)
    traceable_encoder.eval()

    traceable_decoder = TraceableDecoder(model)
    traceable_decoder.eval()

    verify_encoder(model, traceable_encoder)

    # --- Export Encoder ---
    print("\n=== Exporting Encoder to CoreML ===")
    seq_len = 128  # fixed text sequence length for tracing
    input_ids = torch.zeros(1, seq_len, dtype=torch.long)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    bbox = torch.zeros(1, seq_len, 4, dtype=torch.long)
    pixel_values = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    print("  Tracing encoder...")
    with torch.no_grad():
        traced_encoder = torch.jit.trace(
            traceable_encoder,
            (input_ids, attention_mask, bbox, pixel_values),
        )

    print("  Converting to CoreML...")
    enc_out_len = seq_len + NUM_PATCHES  # 128 + 1024 = 1152
    encoder_coreml = ct.convert(
        traced_encoder,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, seq_len), dtype=np.int32),
            ct.TensorType(name="bbox", shape=(1, seq_len, 4), dtype=np.int32),
            ct.TensorType(name="pixel_values", shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="encoder_hidden_states"),
            ct.TensorType(name="encoder_attention_mask"),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    encoder_coreml.short_description = "UDOP Encoder - document image + OCR tokens/boxes -> hidden states"
    encoder_path = os.path.join(OUTPUT_DIR, "UdopEncoder.mlpackage")
    encoder_coreml.save(encoder_path)
    print(f"  Saved to: {encoder_path}")

    # --- Export Decoder ---
    print("\n=== Exporting Decoder to CoreML ===")
    dec_seq_len = 1  # single token for autoregressive generation
    decoder_input_ids = torch.zeros(1, dec_seq_len, dtype=torch.long)
    encoder_hidden_states = torch.randn(1, enc_out_len, model.config.d_model)
    encoder_attention_mask = torch.ones(1, enc_out_len, dtype=torch.long)

    print("  Tracing decoder...")
    with torch.no_grad():
        traced_decoder = torch.jit.trace(
            traceable_decoder,
            (decoder_input_ids, encoder_hidden_states, encoder_attention_mask),
        )

    print("  Converting to CoreML...")
    decoder_coreml = ct.convert(
        traced_decoder,
        inputs=[
            ct.TensorType(name="decoder_input_ids", shape=(1, dec_seq_len), dtype=np.int32),
            ct.TensorType(name="encoder_hidden_states", shape=(1, enc_out_len, model.config.d_model), dtype=np.float32),
            ct.TensorType(name="encoder_attention_mask", shape=(1, enc_out_len), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits"),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    decoder_coreml.short_description = "UDOP Decoder - encoder output + decoder tokens -> logits"
    decoder_path = os.path.join(OUTPUT_DIR, "UdopDecoder.mlpackage")
    decoder_coreml.save(decoder_path)
    print(f"  Saved to: {decoder_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model()
    export_to_coreml(model)

    print("\n=== Done! ===")
    print(f"CoreML models saved to {OUTPUT_DIR}/")
    print(f"  - UdopEncoder.mlpackage (encoder: image + OCR -> hidden states)")
    print(f"  - UdopDecoder.mlpackage (decoder: hidden states -> logits)")
    print(f"\nFor iOS usage:")
    print(f"  1. Pad/truncate OCR tokens to {128} tokens")
    print(f"  2. Run encoder once per document image")
    print(f"  3. Run decoder autoregressively (feed predicted token back)")
    print(f"  4. Use UdopProcessor for preprocessing (tokenization + image normalization)")


if __name__ == "__main__":
    main()
