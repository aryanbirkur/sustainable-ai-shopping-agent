path = 'recommendation/hybrid/__init__.py'
with open(path) as f:
    content = f.read()

old = '''def recommend(
    query: str,
    user_id: Optional[str] = None,
    top_k: int = 10,
    weights: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """
    Blend content, collaborative, and sustainability signals for `query`
    (and `user_id` if known) and return the top_k ranked results.

    Each result dict contains: product_id, final_score, score_breakdown,
    rank, weights_used, cold_start, out_of_domain_query, raw_signals,
    product_name, category, brand, price, sustainability_score.
    """
    candidates = blend(query=query, user_id=user_id, weights=weights)
    if not candidates:
        logger.warning(f"recommend(): no candidates for query='{query}'")
        return []
    return rank(candidates, top_k=top_k)'''

new = '''def recommend(
    query: str,
    user_id: Optional[str] = None,
    top_k: int = 10,
    weights: Optional[Dict[str, float]] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    category: Optional[str] = None,
    sustainability_tilt: bool = False,
) -> List[dict]:
    """
    Blend content, collaborative, and sustainability signals for `query`
    (and `user_id` if known) and return the top_k ranked results.

    Milestone 8 additions (all optional, default to prior behavior):
        price_min, price_max, category, sustainability_tilt -- see blend().

    Each result dict contains: product_id, final_score, score_breakdown,
    rank, weights_used, cold_start, out_of_domain_query, filtering,
    raw_signals, product_name, category, brand, price, sustainability_score.
    """
    candidates = blend(
        query=query,
        user_id=user_id,
        weights=weights,
        price_min=price_min,
        price_max=price_max,
        category=category,
        sustainability_tilt=sustainability_tilt,
    )
    if not candidates:
        logger.warning(f"recommend(): no candidates for query='{query}'")
        return []
    return rank(candidates, top_k=top_k)'''

assert old in content, 'OLD BLOCK NOT FOUND -- aborting, no changes made'
content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)
print('recommend() signature updated')
