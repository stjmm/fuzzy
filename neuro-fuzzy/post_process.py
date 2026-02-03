import torch
import torch.nn.functional as F
import numpy as np


def jaccard_similarity(mu1, sigma1, mu2, sigma2, n_samples=100):
    """
    Compute Jaccard similarity between two Gaussian fuzzy sets
    """
    # Sample points in a reasonable range
    x_min = min(mu1 - 3*sigma1, mu2 - 3*sigma2)
    x_max = max(mu1 + 3*sigma1, mu2 + 3*sigma2)
    x = torch.linspace(x_min, x_max, n_samples)
    
    # Compute membership values
    mem1 = torch.exp(-((x - mu1) / sigma1) ** 2)
    mem2 = torch.exp(-((x - mu2) / sigma2) ** 2)
    
    # Jaccard = intersection / union
    intersection = torch.sum(torch.minimum(mem1, mem2))
    union = torch.sum(torch.maximum(mem1, mem2))
    
    if union > 0:
        return (intersection / union).item()
    return 0.0


def merge_similar_sets(student, threshold=0.92):
    """
    Merge similar fuzzy sets within each input dimension
    Based on Algorithm 3 from the paper
    """
    print(f"\nMerging similar fuzzy sets (threshold={threshold})...")
    
    with torch.no_grad():
        mu = student.mu.squeeze(0).clone()  # [rules, inputs]
        sigma = torch.exp(student.log_sigma).squeeze(0).clone()  # [rules, inputs]
        
        merge_counts = torch.ones(student.num_rules, student.num_inputs)
        merged = True
        total_merges = 0
        
        # Process each input dimension separately
        for input_dim in range(student.num_inputs):
            merged = True
            dim_merges = 0
            
            while merged:
                merged = False
                max_similarity = 0
                best_pair = None
                
                # Find most similar pair of rules for this input dimension
                for r1 in range(student.num_rules):
                    for r2 in range(r1 + 1, student.num_rules):
                        similarity = jaccard_similarity(
                            mu[r1, input_dim],
                            sigma[r1, input_dim],
                            mu[r2, input_dim],
                            sigma[r2, input_dim]
                        )
                        
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_pair = (r1, r2)
                
                # Merge if above threshold
                if max_similarity > threshold and best_pair is not None:
                    r1, r2 = best_pair
                    n1 = merge_counts[r1, input_dim]
                    n2 = merge_counts[r2, input_dim]
                    
                    # Weighted average
                    new_mu = (n1 * mu[r1, input_dim] + n2 * mu[r2, input_dim]) / (n1 + n2)
                    new_sigma = (n1 * sigma[r1, input_dim] + n2 * sigma[r2, input_dim]) / (n1 + n2)
                    
                    # Update both rules to have the same merged set
                    mu[r1, input_dim] = new_mu
                    mu[r2, input_dim] = new_mu
                    sigma[r1, input_dim] = new_sigma
                    sigma[r2, input_dim] = new_sigma
                    
                    # Update merge count
                    merge_counts[r1, input_dim] += merge_counts[r2, input_dim]
                    
                    merged = True
                    dim_merges += 1
                    total_merges += 1
            
            if dim_merges > 0:
                print(f"  Input {input_dim}: Merged {dim_merges} pairs")
        
        # Update model parameters
        student.mu.copy_(mu.unsqueeze(0))
        student.log_sigma.copy_(torch.log(sigma.unsqueeze(0)))
        
        print(f"✓ Total merges: {total_merges}")


def prune_unimportant_weights(student, threshold=0.01):
    """
    Prune importance weights below threshold
    Sets them to zero for interpretability
    """
    print(f"\nPruning unimportant weights (threshold={threshold})...")
    
    with torch.no_grad():
        weights = student.weights.squeeze(0).clone()  # [rules, inputs]
        
        pruned_count = 0
        for rule_idx in range(student.num_rules):
            # Normalize weights for this rule
            max_weight = torch.max(weights[rule_idx])
            
            if max_weight > 0:
                normalized = weights[rule_idx] / max_weight
                
                # Prune weights below threshold
                mask = normalized < threshold
                weights[rule_idx][mask] = 0.0
                pruned_count += mask.sum().item()
        
        # Update model
        student.weights.copy_(weights.unsqueeze(0))
        
        print(f"✓ Pruned {pruned_count} weights")


def optimize_rules(student, threshold=0.92, weight_threshold=0.01):
    """
    Full post-processing: merge similar sets and prune weights
    """
    print("\n" + "="*70)
    print("POST-PROCESSING: Optimizing Rules")
    print("="*70)
    
    merge_similar_sets(student, threshold=threshold)
    prune_unimportant_weights(student, threshold=weight_threshold)
    
    print("\n✓ Post-processing complete")
    print("="*70)
    
    return student


def print_rules(student, state_names=None, action_names=None):
    """
    Print human-readable fuzzy rules
    """
    if state_names is None:
        state_names = ['p_x', 'p_y', 'v_x', 'v_y', 'angle', 'v_a']
    
    if action_names is None:
        action_names = ['Do Nothing', 'Fire Left', 'Fire Main', 'Fire Right']
    
    print("\n" + "="*70)
    print("LEARNED FUZZY RULES")
    print("="*70)
    
    with torch.no_grad():
        mu = student.mu.squeeze(0).cpu().numpy()  # [rules, inputs]
        sigma = torch.exp(student.log_sigma).squeeze(0).cpu().numpy()
        weights = student.weights.squeeze(0).cpu().numpy()
        consequents = student.consequents.cpu().numpy()  # [rules, actions]
    
    for rule_idx in range(student.num_rules):
        # Find max weight for normalization
        max_weight = np.max(weights[rule_idx])
        
        if max_weight > 0:
            normalized_weights = weights[rule_idx] / max_weight
        else:
            normalized_weights = weights[rule_idx]
        
        # Build antecedent (IF part)
        antecedents = []
        for input_idx in range(min(student.num_inputs, len(state_names))):
            w = normalized_weights[input_idx]
            
            # Only include important terms
            if w > 0.01:
                mu_val = mu[rule_idx, input_idx]
                sigma_val = sigma[rule_idx, input_idx]
                
                antecedents.append(
                    f"{state_names[input_idx]:6s} ≈ {mu_val:6.2f} "
                    f"(σ={sigma_val:.2f}, w={w:.2f})"
                )
        
        # Build consequent (THEN part)
        q_values = consequents[rule_idx]
        best_action = np.argmax(q_values)
        q_str = ", ".join([f"{q:6.2f}" for q in q_values])
        
        # Print rule
        print(f"\nRule {rule_idx + 1}:")
        if antecedents:
            print(f"  IF:")
            for ant in antecedents:
                print(f"    {ant}")
        else:
            print(f"  (No active antecedents)")
        
        print(f"  THEN: {action_names[best_action]}")
        print(f"  Q-values: [{q_str}]")
    
    print("\n" + "="*70)


def get_rule_statistics(student):
    """
    Compute statistics about the learned rules
    """
    with torch.no_grad():
        weights = student.weights.squeeze(0).cpu().numpy()
        
        stats = {
            'num_rules': student.num_rules,
            'num_inputs': student.num_inputs,
            'active_terms': 0,
            'avg_terms_per_rule': 0,
            'sparsity': 0
        }
        
        active_count = 0
        for rule_idx in range(student.num_rules):
            max_weight = np.max(weights[rule_idx])
            if max_weight > 0:
                normalized = weights[rule_idx] / max_weight
                active = np.sum(normalized > 0.01)
                active_count += active
        
        stats['active_terms'] = int(active_count)
        stats['avg_terms_per_rule'] = active_count / student.num_rules
        total_possible = student.num_rules * student.num_inputs
        stats['sparsity'] = 1.0 - (active_count / total_possible)
        
        return stats


def analyze_rules(student):
    """
    Analyze and print rule statistics
    """
    print("\n" + "="*70)
    print("RULE ANALYSIS")
    print("="*70)
    
    stats = get_rule_statistics(student)
    
    print(f"Number of Rules: {stats['num_rules']}")
    print(f"Input Dimensions: {stats['num_inputs']}")
    print(f"Active Terms: {stats['active_terms']}/{stats['num_rules'] * stats['num_inputs']}")
    print(f"Avg Terms per Rule: {stats['avg_terms_per_rule']:.2f}")
    print(f"Sparsity: {stats['sparsity']*100:.1f}%")
    
    # Analyze which inputs are most important
    with torch.no_grad():
        weights = student.weights.squeeze(0).cpu().numpy()
        
        # Sum normalized weights across all rules for each input
        importance = np.zeros(student.num_inputs)
        for rule_idx in range(student.num_rules):
            max_weight = np.max(weights[rule_idx])
            if max_weight > 0:
                importance += weights[rule_idx] / max_weight
        
        importance /= student.num_rules
        
        state_names = ['p_x', 'p_y', 'v_x', 'v_y', 'angle', 'v_a']
        print("\nInput Importance (averaged across rules):")
        for i, name in enumerate(state_names[:student.num_inputs]):
            print(f"  {name:8s}: {importance[i]:.3f}")
    
    print("="*70)
