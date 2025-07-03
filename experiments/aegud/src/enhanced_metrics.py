"""
Enhanced Metrics for AEGUD Evaluation
Implements comprehensive metrics for evaluating discrete diffusion models
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json


class EnhancedMetrics:
    """
    Comprehensive metrics for evaluating AEGUD and discrete diffusion models.
    """
    
    def __init__(self, vocab_size: int, device: str = 'cuda'):
        self.vocab_size = vocab_size
        self.device = device
        
        # For tracking metrics over time
        self.metric_history = {
            'semantic_preservation': [],
            'diversity_scores': [],
            'convergence_quality': [],
            'generation_quality': []
        }
    
    def measure_semantic_preservation(self, x_0: torch.Tensor, x_t: torch.Tensor, 
                                    t: float, embeddings: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Quantify how much semantic information survives diffusion.
        
        Args:
            x_0: Original tokens
            x_t: Tokens after diffusion
            t: Time step
            embeddings: Optional pre-computed embeddings
            
        Returns:
            Dictionary of semantic preservation metrics
        """
        metrics = {}
        
        # 1. Token overlap ratio
        x_0_flat = x_0.flatten()
        x_t_flat = x_t.flatten()
        overlap = (x_0_flat == x_t_flat).float().mean()
        metrics['token_overlap'] = overlap.item()
        
        # 2. N-gram preservation
        for n in [2, 3, 4]:
            ngram_score = self._compute_ngram_overlap(x_0, x_t, n)
            metrics[f'{n}gram_overlap'] = ngram_score
        
        # 3. Position-aware similarity
        position_similarity = self._compute_position_similarity(x_0, x_t)
        metrics['position_similarity'] = position_similarity
        
        # 4. Embedding similarity (if embeddings provided)
        if embeddings is not None:
            embed_sim = self._compute_embedding_similarity(x_0, x_t, embeddings)
            metrics['embedding_similarity'] = embed_sim
        
        # 5. Information retention score
        # Higher score means more information preserved
        info_score = np.exp(-t) * overlap.item() + (1 - np.exp(-t)) * 0.1
        metrics['information_retention'] = info_score
        
        return metrics
    
    def _compute_ngram_overlap(self, x_0: torch.Tensor, x_t: torch.Tensor, n: int) -> float:
        """Compute n-gram overlap between sequences."""
        def get_ngrams(seq, n):
            ngrams = []
            for i in range(len(seq) - n + 1):
                ngrams.append(tuple(seq[i:i+n].tolist()))
            return Counter(ngrams)
        
        overlap_scores = []
        for i in range(x_0.shape[0]):
            ngrams_0 = get_ngrams(x_0[i], n)
            ngrams_t = get_ngrams(x_t[i], n)
            
            if len(ngrams_0) == 0:
                overlap_scores.append(0.0)
                continue
            
            common = sum((ngrams_0 & ngrams_t).values())
            total = sum(ngrams_0.values())
            overlap_scores.append(common / total)
        
        return np.mean(overlap_scores)
    
    def _compute_position_similarity(self, x_0: torch.Tensor, x_t: torch.Tensor) -> float:
        """Compute position-weighted similarity."""
        batch_size, seq_len = x_0.shape
        
        # Weight positions (beginning and end more important)
        position_weights = torch.ones(seq_len, device=x_0.device)
        position_weights[:seq_len//4] = 2.0  # Beginning
        position_weights[-seq_len//4:] = 1.5  # End
        
        # Compute weighted similarity
        matches = (x_0 == x_t).float()
        weighted_matches = matches * position_weights.unsqueeze(0)
        
        return weighted_matches.sum().item() / (batch_size * position_weights.sum().item())
    
    def _compute_embedding_similarity(self, x_0: torch.Tensor, x_t: torch.Tensor, 
                                    embeddings: torch.Tensor) -> float:
        """Compute cosine similarity in embedding space."""
        # Get embeddings for sequences
        emb_0 = embeddings[x_0]  # (batch, seq_len, embed_dim)
        emb_t = embeddings[x_t]
        
        # Compute sequence-level embeddings (mean pooling)
        seq_emb_0 = emb_0.mean(dim=1)  # (batch, embed_dim)
        seq_emb_t = emb_t.mean(dim=1)
        
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(seq_emb_0, seq_emb_t, dim=1)
        
        return cos_sim.mean().item()
    
    def analyze_generation_diversity(self, samples: List[torch.Tensor], 
                                   num_buckets: int = 10) -> Dict[str, float]:
        """
        Analyze diversity of generated samples.
        
        Args:
            samples: List of generated sequences
            num_buckets: Number of buckets for entropy calculation
            
        Returns:
            Dictionary of diversity metrics
        """
        metrics = {}
        
        # Convert to numpy for easier processing
        samples_np = [s.cpu().numpy() for s in samples]
        all_samples = np.concatenate(samples_np, axis=0)
        
        # 1. Unique n-grams
        for n in [1, 2, 3, 4]:
            unique_ngrams = set()
            total_ngrams = 0
            
            for sample in samples_np:
                for seq in sample:
                    for i in range(len(seq) - n + 1):
                        ngram = tuple(seq[i:i+n])
                        unique_ngrams.add(ngram)
                        total_ngrams += 1
            
            metrics[f'unique_{n}grams_ratio'] = len(unique_ngrams) / max(1, total_ngrams)
        
        # 2. Self-BLEU (lower is more diverse)
        self_bleu = self._compute_self_bleu(samples_np)
        metrics['self_bleu'] = self_bleu
        
        # 3. Token distribution entropy
        token_counts = Counter(all_samples.flatten())
        total_tokens = sum(token_counts.values())
        token_probs = np.array([token_counts[i] / total_tokens 
                               for i in range(self.vocab_size)])
        token_entropy = -np.sum(token_probs * np.log(token_probs + 1e-10))
        metrics['token_entropy'] = token_entropy / np.log(self.vocab_size)  # Normalized
        
        # 4. Sequence diversity score
        unique_sequences = len(set(tuple(seq) for sample in samples_np for seq in sample))
        total_sequences = sum(len(sample) for sample in samples_np)
        metrics['unique_sequence_ratio'] = unique_sequences / max(1, total_sequences)
        
        # 5. Perplexity estimate (based on token distribution)
        metrics['estimated_perplexity'] = np.exp(token_entropy)
        
        return metrics
    
    def _compute_self_bleu(self, samples: List[np.ndarray], max_n: int = 4) -> float:
        """Compute Self-BLEU score."""
        # Simplified Self-BLEU calculation
        # In practice, would use proper BLEU scoring
        
        all_sequences = []
        for sample in samples:
            for seq in sample:
                all_sequences.append(seq)
        
        if len(all_sequences) < 2:
            return 0.0
        
        # Compute average BLEU score of each sequence against all others
        bleu_scores = []
        
        for i, seq in enumerate(all_sequences[:100]):  # Limit for efficiency
            # Create reference set (all sequences except current)
            refs = all_sequences[:i] + all_sequences[i+1:]
            
            # Compute BLEU-like score (simplified)
            score = 0.0
            for n in range(1, min(max_n + 1, len(seq) + 1)):
                seq_ngrams = Counter(tuple(seq[j:j+n]) for j in range(len(seq) - n + 1))
                
                ref_ngrams = Counter()
                for ref in refs[:50]:  # Limit references for efficiency
                    for j in range(len(ref) - n + 1):
                        ref_ngrams[tuple(ref[j:j+n])] += 1
                
                # Compute precision
                overlap = sum((seq_ngrams & ref_ngrams).values())
                total = sum(seq_ngrams.values())
                
                if total > 0:
                    precision = overlap / total
                    score += precision / max_n
            
            bleu_scores.append(score)
        
        return np.mean(bleu_scores)
    
    def measure_convergence_quality(self, graph, noise, num_steps: int = 100,
                                  batch_size: int = 32, seq_len: int = 64) -> Dict[str, any]:
        """
        Comprehensive convergence quality measurement.
        """
        device = next(graph.parameters()).device if hasattr(graph, 'parameters') else self.device
        
        # Generate random initial sequences
        x_0 = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=device)
        
        convergence_metrics = {
            'entropy_trajectory': [],
            'kl_trajectory': [],
            'chi2_pvalues': [],
            'mutual_information': [],
            'effective_vocab_size': [],
            'convergence_rate': 0.0
        }
        
        # Track metrics over diffusion process
        for step in range(num_steps + 1):
            t = step / num_steps
            
            # Get noise level
            sigma, _ = noise(torch.tensor([t], device=device))
            
            # Apply diffusion
            x_t = graph.sample_transition(x_0, sigma.expand(batch_size, 1))
            
            # Compute token distribution
            token_counts = torch.zeros(self.vocab_size, device=device)
            x_t_flat = x_t.flatten()
            token_counts.scatter_add_(0, x_t_flat, torch.ones_like(x_t_flat, dtype=torch.float))
            token_probs = token_counts / (batch_size * seq_len)
            
            # 1. Entropy
            entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-10))
            normalized_entropy = entropy / np.log(self.vocab_size)
            convergence_metrics['entropy_trajectory'].append(normalized_entropy.item())
            
            # 2. KL divergence from uniform
            uniform_probs = torch.ones_like(token_probs) / self.vocab_size
            kl_div = F.kl_div(uniform_probs.log(), token_probs, reduction='sum')
            convergence_metrics['kl_trajectory'].append(kl_div.item())
            
            # 3. Chi-squared test
            expected_count = (batch_size * seq_len) / self.vocab_size
            observed_counts = token_counts.cpu().numpy()
            chi2_stat, p_value = stats.chisquare(observed_counts, 
                                                f_exp=[expected_count] * self.vocab_size)
            convergence_metrics['chi2_pvalues'].append(p_value)
            
            # 4. Mutual information estimate
            if hasattr(graph, 'entropy_estimator'):
                info_content = graph.entropy_estimator.estimate_information_content(x_t)
                mi_estimate = info_content.mean().item() * (1 - t)
            else:
                # Simple estimate based on token overlap
                overlap = (x_0 == x_t).float().mean()
                mi_estimate = overlap.item() * np.log(self.vocab_size) * (1 - t)
            convergence_metrics['mutual_information'].append(mi_estimate)
            
            # 5. Effective vocabulary size (perplexity)
            perplexity = torch.exp(entropy)
            convergence_metrics['effective_vocab_size'].append(perplexity.item())
        
        # Compute convergence rate (how fast entropy approaches maximum)
        entropy_traj = np.array(convergence_metrics['entropy_trajectory'])
        if len(entropy_traj) > 10:
            # Fit exponential to entropy trajectory
            t_values = np.linspace(0, 1, len(entropy_traj))
            try:
                # Fit y = 1 - exp(-rate * t)
                from scipy.optimize import curve_fit
                def exp_func(t, rate):
                    return 1 - np.exp(-rate * t)
                
                popt, _ = curve_fit(exp_func, t_values[1:], entropy_traj[1:], p0=[1.0])
                convergence_metrics['convergence_rate'] = popt[0]
            except:
                convergence_metrics['convergence_rate'] = 0.0
        
        # Final convergence status
        final_entropy = convergence_metrics['entropy_trajectory'][-1]
        final_kl = convergence_metrics['kl_trajectory'][-1]
        final_chi2_p = convergence_metrics['chi2_pvalues'][-1]
        
        convergence_metrics['final_status'] = {
            'entropy_converged': final_entropy > 0.95,
            'kl_converged': final_kl < 0.01,
            'chi2_converged': final_chi2_p > 0.05,
            'overall_converged': final_entropy > 0.95 and final_kl < 0.01
        }
        
        return convergence_metrics
    
    def create_convergence_visualization(self, convergence_metrics: Dict[str, any],
                                       save_path: str = None) -> None:
        """
        Create comprehensive visualization of convergence metrics.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Convergence Analysis', fontsize=16)
        
        t_values = np.linspace(0, 1, len(convergence_metrics['entropy_trajectory']))
        
        # 1. Entropy trajectory
        ax = axes[0, 0]
        ax.plot(t_values, convergence_metrics['entropy_trajectory'], 'b-', linewidth=2)
        ax.axhline(y=1.0, color='r', linestyle='--', label='Target')
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Normalized Entropy')
        ax.set_title('Entropy Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. KL divergence trajectory
        ax = axes[0, 1]
        ax.semilogy(t_values, convergence_metrics['kl_trajectory'], 'g-', linewidth=2)
        ax.axhline(y=0.01, color='r', linestyle='--', label='Target')
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('KL Divergence (log scale)')
        ax.set_title('KL from Uniform')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Chi-squared p-values
        ax = axes[0, 2]
        ax.plot(t_values, convergence_metrics['chi2_pvalues'], 'm-', linewidth=2)
        ax.axhline(y=0.05, color='r', linestyle='--', label='α = 0.05')
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('p-value')
        ax.set_title('Chi-squared Test')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Mutual information
        ax = axes[1, 0]
        ax.plot(t_values, convergence_metrics['mutual_information'], 'c-', linewidth=2)
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Mutual Information')
        ax.set_title('Information Decay')
        ax.grid(True, alpha=0.3)
        
        # 5. Effective vocabulary size
        ax = axes[1, 1]
        ax.plot(t_values, convergence_metrics['effective_vocab_size'], 'orange', linewidth=2)
        ax.axhline(y=self.vocab_size, color='r', linestyle='--', label='Max')
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Effective Vocab Size')
        ax.set_title('Perplexity Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Convergence summary
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = "Final Convergence Status:\n\n"
        status = convergence_metrics['final_status']
        for key, value in status.items():
            status_str = "✓ PASS" if value else "✗ FAIL"
            summary_text += f"{key.replace('_', ' ').title()}: {status_str}\n"
        
        summary_text += f"\nConvergence Rate: {convergence_metrics['convergence_rate']:.3f}"
        
        ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def compute_scale_dependent_metrics(self, graph, noise, 
                                      vocab_sizes: List[int] = [50, 500, 5000],
                                      diffusion_steps: List[int] = [50, 100, 500]) -> Dict[str, any]:
        """
        Analyze how metrics change with scale.
        """
        results = {}
        
        for vocab_size in vocab_sizes:
            for steps in diffusion_steps:
                key = f"vocab_{vocab_size}_steps_{steps}"
                
                # Create temporary graph with different vocab size
                # Note: This is simplified - in practice would need proper initialization
                metrics = self.measure_convergence_quality(
                    graph, noise, 
                    num_steps=steps,
                    batch_size=16,
                    seq_len=32
                )
                
                results[key] = {
                    'final_entropy': metrics['entropy_trajectory'][-1],
                    'final_kl': metrics['kl_trajectory'][-1],
                    'convergence_rate': metrics['convergence_rate'],
                    'converged': metrics['final_status']['overall_converged']
                }
        
        return results
    
    def save_metrics(self, filepath: str) -> None:
        """Save all collected metrics to file."""
        with open(filepath, 'w') as f:
            json.dump(self.metric_history, f, indent=2)
    
    def load_metrics(self, filepath: str) -> None:
        """Load previously saved metrics."""
        with open(filepath, 'r') as f:
            self.metric_history = json.load(f)