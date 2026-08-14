def _aggregate_results(self, results: List[Dict], successful_models: List[str]) -> Dict:
    """Aggregate results from ALL models - count all picks across all positions."""
    if not results:
        return {"selections": [], "consensus": {}}
    
    # Count votes per ticker (across all positions)
    ticker_votes = {}
    ticker_data = {}
    total_responses = len(results)
    
    for r in results:
        ticker = r.get("ticker", "").upper()
        if ticker:
            if ticker not in ticker_votes:
                ticker_votes[ticker] = 0
                ticker_data[ticker] = {
                    "returns": [],
                    "confidences": [],
                    "rationales": [],
                    "models": []
                }
            ticker_votes[ticker] += 1  # Each selection counts as 1 vote
            ticker_data[ticker]["returns"].append(r.get("expected_return", 0.5))
            ticker_data[ticker]["confidences"].append(r.get("confidence", "Medium"))
            ticker_data[ticker]["rationales"].append(r.get("rationale", ""))
            ticker_data[ticker]["models"].append(r.get("model", "unknown"))
    
    # Sort by votes (descending), then by average return
    sorted_tickers = sorted(
        ticker_votes.items(),
        key=lambda x: (x[1], sum(ticker_data[x[0]]["returns"]) / len(ticker_data[x[0]]["returns"])),
        reverse=True
    )
    
    # Take top N
    top_tickers = [t for t, _ in sorted_tickers[:self.top_n]]
    
    selections = []
    for ticker in top_tickers:
        data = ticker_data[ticker]
        avg_return = sum(data["returns"]) / len(data["returns"])
        
        # Most common confidence
        conf_counts = {}
        for c in data["confidences"]:
            conf_counts[c] = conf_counts.get(c, 0) + 1
        top_conf = max(conf_counts, key=conf_counts.get)
        
        # Unique models that voted for this ticker
        unique_models = list(set([m for m in data["models"] if m and m != 'unknown']))
        
        selections.append({
            "ticker": ticker,
            "expected_return": round(avg_return, 2),
            "confidence": top_conf,
            "rationale": data["rationales"][0] if data["rationales"] else "",
            "votes": ticker_votes[ticker],  # Number of times this ticker was picked (across all positions)
            "models": unique_models
        })
    
    # All unique models used
    all_models_used = list(set(successful_models))
    
    return {
        "selections": selections,
        "consensus": {
            "total_votes": total_responses,
            "ticker_votes": ticker_votes,
            "models_used": all_models_used
        }
    }
