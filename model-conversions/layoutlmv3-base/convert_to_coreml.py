"""
Convert LayoutLMv3ForTokenClassification (nnul/layoutlmv3-finetuned-funsd) to CoreML.

This is a FUNSD fine-tuned LayoutLMv3 for token classification (NER) on forms:
  Labels: O, B-HEADER, I-HEADER, B-QUESTION, I-QUESTION, B-ANSWER, I-ANSWER (7 classes)

Architecture (encoder-only, RoBERTa-style):
  - Text embeddings: word + position + token_type + spatial (bbox x/y/w/h)
  - Image embeddings: Conv2d patches (14x14=196) + [CLS] + position → 197 visual tokens
  - Text + visual concatenated → shared transformer encoder (12 layers)
  - Relative position biases: 1D positional + 2D spatial (bbox-based)
  - CogView attention: numerically-stable softmax variant
  - Token classification head: slices text tokens, dropout + Linear → 7 logits

CoreML model:
  Inputs:  input_ids [1, 512], attention_mask [1, 512], bbox [1, 512, 4], pixel_values [1, 3, 224, 224]
  Outputs: logits [1, 512, 7]
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np

MODEL_DIR = "model-hf"
OUTPUT_DIR = "output"
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES_PER_DIM = IMAGE_SIZE // PATCH_SIZE  # 14
NUM_PATCHES = NUM_PATCHES_PER_DIM ** 2  # 196
NUM_VISUAL_TOKENS = NUM_PATCHES + 1  # 197 (196 patches + CLS)
TEXT_SEQ_LEN = 512
NUM_LABELS = 7  # FUNSD: O, B-HEADER, I-HEADER, B-QUESTION, I-QUESTION, B-ANSWER, I-ANSWER


def load_model():
    from transformers import LayoutLMv3ForTokenClassification
    print("Loading LayoutLMv3ForTokenClassification model...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    model.eval()
    return model


def precompute_visual_bbox(image_size=14, max_len=1000):
    """Pre-compute bounding boxes for visual (patch) tokens, matching init_visual_bbox."""
    visual_bbox_x = torch.div(
        torch.arange(0, max_len * (image_size + 1), max_len), image_size, rounding_mode="trunc"
    )
    visual_bbox_y = torch.div(
        torch.arange(0, max_len * (image_size + 1), max_len), image_size, rounding_mode="trunc"
    )
    visual_bbox = torch.stack(
        [
            visual_bbox_x[:-1].repeat(image_size, 1),
            visual_bbox_y[:-1].repeat(image_size, 1).transpose(0, 1),
            visual_bbox_x[1:].repeat(image_size, 1),
            visual_bbox_y[1:].repeat(image_size, 1).transpose(0, 1),
        ],
        dim=-1,
    ).view(-1, 4)

    cls_token_box = torch.tensor([[1, 1, max_len - 1, max_len - 1]])
    return torch.cat([cls_token_box, visual_bbox], dim=0)  # [197, 4]


def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    """Log-scale bucketing for relative positions (bidirectional)."""
    half_buckets = num_buckets // 2
    ret = (relative_position > 0).long() * half_buckets
    n = torch.abs(relative_position)

    max_exact = half_buckets // 2
    is_small = n < max_exact

    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (half_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, half_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret


class TraceableLayoutLMv3ForTokenClassification(nn.Module):
    """Fully traceable LayoutLMv3 + token classification head for CoreML conversion.

    Flattens all conditional branches, pre-computes visual bboxes and position IDs,
    inlines relative position bias computation, and includes the classifier head.
    Output: logits [B, TEXT_SEQ_LEN, NUM_LABELS] (only text token predictions).
    """
    def __init__(self, model):
        super().__init__()
        base = model.layoutlmv3
        self.config = base.config

        # Text embeddings
        self.word_embeddings = base.embeddings.word_embeddings
        self.token_type_embeddings = base.embeddings.token_type_embeddings
        self.position_embeddings = base.embeddings.position_embeddings
        self.x_position_embeddings = base.embeddings.x_position_embeddings
        self.y_position_embeddings = base.embeddings.y_position_embeddings
        self.h_position_embeddings = base.embeddings.h_position_embeddings
        self.w_position_embeddings = base.embeddings.w_position_embeddings
        self.embed_layernorm = base.embeddings.LayerNorm
        self.embed_dropout = base.embeddings.dropout

        # Visual embeddings
        self.patch_embed = base.patch_embed
        self.cls_token = base.cls_token
        self.pos_embed = base.pos_embed
        self.pos_drop = base.pos_drop
        self.visual_norm = base.norm

        # Combined LayerNorm + dropout (after text+visual concat)
        self.combined_layernorm = base.LayerNorm
        self.combined_dropout = base.dropout

        # Encoder
        self.encoder = base.encoder

        # Token classification head
        self.cls_dropout = model.dropout
        self.classifier = model.classifier

        # Pre-compute visual bbox as buffer
        visual_bbox = precompute_visual_bbox()  # [197, 4]
        self.register_buffer("visual_bbox", visual_bbox)

        # Pre-compute 1D relative position bias for the full sequence
        # position_ids: text uses RoBERTa-style (cumsum), visual=[0..196]
        # For the 1D bias, we use position indices directly
        text_position_ids = torch.arange(0, TEXT_SEQ_LEN, dtype=torch.long)
        visual_position_ids = torch.arange(0, NUM_VISUAL_TOKENS, dtype=torch.long)
        full_position_ids = torch.cat([text_position_ids, visual_position_ids])

        rel_pos_mat = full_position_ids.unsqueeze(-1) - full_position_ids.unsqueeze(0)
        rel_pos_buckets = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.config.rel_pos_bins,
            max_distance=self.config.max_rel_pos,
        )
        with torch.no_grad():
            rel_pos_bias = self.encoder.rel_pos_bias.weight.t()[rel_pos_buckets]
            rel_pos_bias = rel_pos_bias.permute(2, 0, 1).unsqueeze(0)
        self.register_buffer("rel_1d_pos", rel_pos_bias.contiguous())

    def forward(self, input_ids, attention_mask, bbox, pixel_values):
        batch_size = input_ids.shape[0]

        # === Text Embeddings ===
        padding_idx = self.config.pad_token_id
        mask = (input_ids != padding_idx).to(torch.long)
        position_ids = torch.cumsum(mask, dim=1) * mask + padding_idx

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        position_embeds = self.position_embeddings(position_ids)

        # Spatial embeddings from bbox
        left_emb = self.x_position_embeddings(bbox[:, :, 0])
        upper_emb = self.y_position_embeddings(bbox[:, :, 1])
        right_emb = self.x_position_embeddings(bbox[:, :, 2])
        lower_emb = self.y_position_embeddings(bbox[:, :, 3])
        h_emb = self.h_position_embeddings(torch.clamp(bbox[:, :, 3] - bbox[:, :, 1], 0, 1023))
        w_emb = self.w_position_embeddings(torch.clamp(bbox[:, :, 2] - bbox[:, :, 0], 0, 1023))
        spatial_emb = torch.cat([left_emb, upper_emb, right_emb, lower_emb, h_emb, w_emb], dim=-1)

        text_embeddings = inputs_embeds + token_type_embeds + position_embeds + spatial_emb
        text_embeddings = self.embed_layernorm(text_embeddings)
        text_embeddings = self.embed_dropout(text_embeddings)

        # === Visual Embeddings ===
        patch_embeddings = self.patch_embed(pixel_values)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        visual_embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)
        visual_embeddings = visual_embeddings + self.pos_embed
        visual_embeddings = self.pos_drop(visual_embeddings)
        visual_embeddings = self.visual_norm(visual_embeddings)

        # === Concatenate text + visual ===
        embedding_output = torch.cat([text_embeddings, visual_embeddings], dim=1)
        embedding_output = self.combined_layernorm(embedding_output)
        embedding_output = self.combined_dropout(embedding_output)

        # === Attention mask ===
        visual_attention_mask = torch.ones(batch_size, NUM_VISUAL_TOKENS, dtype=attention_mask.dtype,
                                           device=attention_mask.device)
        full_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
        extended_attention_mask = full_attention_mask[:, None, None, :].to(dtype=embedding_output.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(embedding_output.dtype).min

        # === 2D spatial position bias ===
        visual_bbox = self.visual_bbox.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=torch.long)
        full_bbox = torch.cat([bbox, visual_bbox], dim=1)

        position_coord_x = full_bbox[:, :, 0]
        position_coord_y = full_bbox[:, :, 3]
        rel_pos_x_mat = position_coord_x.unsqueeze(-1) - position_coord_x.unsqueeze(-2)
        rel_pos_y_mat = position_coord_y.unsqueeze(-1) - position_coord_y.unsqueeze(-2)

        rel_pos_x = relative_position_bucket(
            rel_pos_x_mat,
            num_buckets=self.config.rel_2d_pos_bins,
            max_distance=self.config.max_rel_2d_pos,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_mat,
            num_buckets=self.config.rel_2d_pos_bins,
            max_distance=self.config.max_rel_2d_pos,
        )

        rel_pos_x_bias = self.encoder.rel_pos_x_bias.weight.t()[rel_pos_x].permute(0, 3, 1, 2)
        rel_pos_y_bias = self.encoder.rel_pos_y_bias.weight.t()[rel_pos_y].permute(0, 3, 1, 2)
        rel_2d_pos = (rel_pos_x_bias + rel_pos_y_bias).contiguous()

        rel_1d_pos = self.rel_1d_pos.expand(batch_size, -1, -1, -1)

        # === Run encoder layers ===
        hidden_states = embedding_output
        for layer_module in self.encoder.layer:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=None,
                output_attentions=False,
                rel_pos=rel_1d_pos,
                rel_2d_pos=rel_2d_pos,
            )
            hidden_states = layer_outputs[0]

        # === Token classification head (text tokens only) ===
        text_hidden_states = hidden_states[:, :TEXT_SEQ_LEN, :]
        text_hidden_states = self.cls_dropout(text_hidden_states)
        logits = self.classifier(text_hidden_states)

        return logits


def verify_model(model, traceable_model):
    """Verify the traceable model produces equivalent outputs to the original."""
    from transformers import LayoutLMv3Processor
    from PIL import Image

    print("\n=== Verifying Traceable Model ===")
    processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR, apply_ocr=False)

    img = Image.new('RGB', (224, 224), color='white')
    words = ['Hello', 'world', 'test']
    boxes = [[100, 100, 200, 200], [300, 300, 400, 400], [500, 500, 600, 600]]
    encoding = processor(images=img, text=words, boxes=boxes, return_tensors='pt',
                         padding='max_length', max_length=TEXT_SEQ_LEN, truncation=True)

    with torch.no_grad():
        orig_out = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            bbox=encoding['bbox'],
            pixel_values=encoding['pixel_values'],
            return_dict=True,
        )

        trace_out = traceable_model(
            encoding['input_ids'],
            encoding['attention_mask'],
            encoding['bbox'],
            encoding['pixel_values'],
        )

    print(f"  Original logits shape: {orig_out.logits.shape}")
    print(f"  Traceable logits shape: {trace_out.shape}")

    cos_sim = torch.nn.functional.cosine_similarity(
        orig_out.logits.flatten(),
        trace_out.flatten(),
        dim=0,
    )
    max_diff = (orig_out.logits - trace_out).abs().max().item()
    print(f"  Cosine similarity: {cos_sim.item():.6f}")
    print(f"  Max absolute diff: {max_diff:.6e}")

    # Check that argmax predictions match
    orig_preds = orig_out.logits.argmax(dim=-1)
    trace_preds = trace_out.argmax(dim=-1)
    match_pct = (orig_preds == trace_preds).float().mean().item() * 100
    print(f"  Prediction match: {match_pct:.1f}%")

    if cos_sim.item() > 0.999:
        print("  PASS: Outputs match!")
    else:
        print("  WARNING: Outputs diverge - check the traceable model")


def export_to_coreml(model):
    """Export LayoutLMv3ForTokenClassification to CoreML via tracing."""
    import coremltools as ct

    traceable_model = TraceableLayoutLMv3ForTokenClassification(model)
    traceable_model.eval()

    verify_model(model, traceable_model)

    print("\n=== Exporting LayoutLMv3 to CoreML ===")

    input_ids = torch.zeros(1, TEXT_SEQ_LEN, dtype=torch.long)
    attention_mask = torch.ones(1, TEXT_SEQ_LEN, dtype=torch.long)
    bbox = torch.zeros(1, TEXT_SEQ_LEN, 4, dtype=torch.long)
    pixel_values = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    print("  Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(
            traceable_model,
            (input_ids, attention_mask, bbox, pixel_values),
        )

    print("  Converting to CoreML...")
    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, TEXT_SEQ_LEN), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, TEXT_SEQ_LEN), dtype=np.int32),
            ct.TensorType(name="bbox", shape=(1, TEXT_SEQ_LEN, 4), dtype=np.int32),
            ct.TensorType(name="pixel_values", shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="logits"),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )

    coreml_model.short_description = (
        "LayoutLMv3 Token Classification (FUNSD) - document image + OCR tokens/boxes -> NER logits"
    )

    output_path = os.path.join(OUTPUT_DIR, "LayoutLMv3.mlpackage")
    coreml_model.save(output_path)
    print(f"  Saved to: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model()
    export_to_coreml(model)

    label_names = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
    print("\n=== Done! ===")
    print(f"CoreML model saved to {OUTPUT_DIR}/")
    print(f"  - LayoutLMv3.mlpackage")
    print(f"\nModel details:")
    print(f"  Inputs:  input_ids [1, {TEXT_SEQ_LEN}], attention_mask [1, {TEXT_SEQ_LEN}],")
    print(f"           bbox [1, {TEXT_SEQ_LEN}, 4], pixel_values [1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}]")
    print(f"  Output:  logits [1, {TEXT_SEQ_LEN}, {NUM_LABELS}]")
    print(f"  Labels:  {label_names}")
    print(f"\nFor iOS usage:")
    print(f"  1. Add LayoutLMv3.mlpackage to Xcode target (Copy Bundle Resources)")
    print(f"  2. Add vocab.json from model-hf/ to the app bundle")
    print(f"  3. Use LayoutLMv3Processor with apply_ocr=False for preprocessing")
    print(f"  4. Pad/truncate OCR tokens to {TEXT_SEQ_LEN}")
    print(f"  5. argmax(logits, dim=-1) gives per-token label predictions")


if __name__ == "__main__":
    main()
