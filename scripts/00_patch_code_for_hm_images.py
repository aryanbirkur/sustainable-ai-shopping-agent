#!/usr/bin/env python3
"""
scripts/patch_code_for_hm_images.py

Applies the 4 source edits needed to carry `image_path` from raw data
all the way to the Streamlit UI. Idempotent -- safe to re-run; each
edit is skipped (with a message) if it's already applied, and the
script exits with an error (without touching anything else) if a file
doesn't match what's expected, rather than guessing where to patch.

Run from your project root, BEFORE scripts/integrate_hm_data.py's
build_vector_index step:
    python scripts/patch_code_for_hm_images.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def apply_edit(rel_path: str, old: str, new: str, already_applied_marker: str):
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        print(f"  SKIP (file not found): {rel_path}")
        return False

    text = path.read_text(encoding="utf-8")

    if already_applied_marker in text:
        print(f"  already patched: {rel_path}")
        return True

    if old not in text:
        print(f"  ERROR: expected text not found in {rel_path} -- not modified.")
        print(f"         (file may have changed since this patch was written; edit it by hand)")
        return False

    if text.count(old) > 1:
        print(f"  ERROR: expected text appears more than once in {rel_path} -- not modified "
              f"(ambiguous match, edit by hand).")
        return False

    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"  patched: {rel_path}")
    return True


def main():
    print("1. scripts/build_vector_index.py -- store image_path in Chroma metadata")
    ok1 = apply_edit(
        "scripts/build_vector_index.py",
        old='REQUIRED_METADATA_FIELDS = ["product_id", "category", "price", "sustainability_score"]',
        new='REQUIRED_METADATA_FIELDS = ["product_id", "category", "price", "sustainability_score", "image_path"]',
        already_applied_marker='"sustainability_score", "image_path"',
    )

    print("\n2. recommendation/content_based/content_scorer.py -- carry image_path into blender metadata")
    ok2 = apply_edit(
        "recommendation/content_based/content_scorer.py",
        old='''metadata[pid] = {
            "product_name": m.get("product_name"),
            "category": m.get("category"),
            "brand": m.get("brand"),
            "price": m.get("price"),
            "sustainability_score": m.get("sustainability_score"),
        }''',
        new='''metadata[pid] = {
            "product_name": m.get("product_name"),
            "category": m.get("category"),
            "brand": m.get("brand"),
            "price": m.get("price"),
            "sustainability_score": m.get("sustainability_score"),
            "image_path": m.get("image_path"),
        }''',
        already_applied_marker='"image_path": m.get("image_path")',
    )

    print("\n3. app/schemas.py -- add image_path to the ResultItem response model")
    ok3 = apply_edit(
        "app/schemas.py",
        old="    price: float\n    sustainability_score: float",
        new="    price: float\n    sustainability_score: float\n    image_path: Optional[str] = None",
        already_applied_marker="image_path: Optional[str] = None",
    )

    print("\n4. frontend/components.py -- render the product image in the result card")
    ok4 = apply_edit(
        "frontend/components.py",
        old='''def render_result_card(st, item):
    badge_class = sustainability_badge_class(item["sustainability_score"])
    st.markdown(''',
        new='''def render_result_card(st, item):
    badge_class = sustainability_badge_class(item["sustainability_score"])
    if item.get("image_path"):
        st.image(item["image_path"], use_container_width=True)
    st.markdown(''',
        already_applied_marker='if item.get("image_path"):',
    )

    if all([ok1, ok2, ok3, ok4]):
        print("\nAll 4 patches applied successfully.")
    else:
        print("\nOne or more patches need manual attention -- see ERROR lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
