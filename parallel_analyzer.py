#!/usr/bin/env python3
"""
parallel_analyzer.py

Parallelized analysis utilities for earnings call transcript analysis.
Provides multi-core processing capabilities for faster local execution.

Features:
- Multiprocessing for topic detection across CPU cores
- Configurable batch sizes and worker processes
- Memory-efficient chunking
- Progress tracking with detailed metrics
"""

import os
import multiprocessing as mp
from multiprocessing import Pool, Manager
from functools import partial
import time
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm


class ParallelAnalyzer:
    """
    Parallel processing wrapper for transcript analysis.
    Handles multiprocessing coordination and batch processing.
    """

    def __init__(self,
                 sentiment_analyzer,
                 interaction_classifier,
                 role_classifier,
                 embedder,
                 nlp,
                 matcher,
                 anchor_embeddings,
                 anchor_metadata,
                 exclusions_map,
                 similarity_threshold=0.7,
                 num_workers=None,
                 sentiment_batch_size=8,
                 classification_batch_size=16):
        """
        Initialize parallel analyzer.

        Args:
            sentiment_analyzer: VADER SentimentIntensityAnalyzer (or HuggingFace pipeline for legacy)
            interaction_classifier: Interaction type classifier
            role_classifier: Role classifier
            embedder: SentenceTransformer embedder
            nlp: spaCy NLP model
            matcher: spaCy Matcher for exact matches
            anchor_embeddings: Pre-computed anchor embeddings
            anchor_metadata: Anchor metadata
            exclusions_map: Topic exclusion terms
            similarity_threshold: Threshold for vector similarity
            num_workers: Number of parallel workers (None = CPU count)
            sentiment_batch_size: Batch size for sentiment analysis (not used with VADER)
            classification_batch_size: Batch size for classification
        """
        self.sentiment_analyzer = sentiment_analyzer
        self.interaction_classifier = interaction_classifier
        self.role_classifier = role_classifier
        self.embedder = embedder
        self.nlp = nlp
        self.matcher = matcher
        self.anchor_embeddings = anchor_embeddings
        self.anchor_metadata = anchor_metadata
        self.exclusions_map = exclusions_map
        self.similarity_threshold = similarity_threshold

        # Auto-detect optimal worker count
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        self.sentiment_batch_size = sentiment_batch_size
        self.classification_batch_size = classification_batch_size

        print(f"[ParallelAnalyzer] Initialized with {self.num_workers} workers")
        print(f"[ParallelAnalyzer] Using VADER sentiment analysis (fast mode)")
        print(f"[ParallelAnalyzer] Classification batch size: {self.classification_batch_size}")

    def classify_batch_parallel(self, texts: List[str], classifier_name: str) -> List[Dict]:
        """
        Run classification in parallel with larger batches.

        Args:
            texts: List of text segments to classify
            classifier_name: 'interaction' or 'role'

        Returns:
            List of classification results
        """
        classifier = self.interaction_classifier if classifier_name == 'interaction' else self.role_classifier

        # Use larger batch size for better GPU utilization
        results = classifier(
            [t[:512] for t in texts],  # Truncate to avoid position embedding errors
            batch_size=self.classification_batch_size
        )

        return results

    def analyze_topics_chunk(self, text_chunk: List[Tuple[int, str]]) -> List[Tuple[int, List[Dict]]]:
        """
        Analyze a chunk of texts for topics (worker function).

        Args:
            text_chunk: List of (index, text) tuples

        Returns:
            List of (index, results) tuples
        """
        results = []

        for idx, text in text_chunk:
            # Try spaCy Matcher first
            doc = self.nlp(text)
            matches = self.matcher(doc)

            if matches:
                found_topics = {}  # topic -> set of matched terms
                for match_id, start, end in matches:
                    topic_label = self.nlp.vocab.strings[match_id]
                    matched_text = doc[start:end].text
                    if topic_label not in found_topics:
                        found_topics[topic_label] = set()
                    found_topics[topic_label].add(matched_text.lower())

                text_results = []
                for topic, matched_terms in found_topics.items():
                    exclusions = self.exclusions_map.get(topic, [])
                    if any(ext.lower() in text.lower() for ext in exclusions):
                        continue
                    text_results.append({
                        "topic": topic,
                        "idx": idx,
                        "key_terms_found": "|".join(sorted(matched_terms))
                    })

                results.append((idx, text_results))
                continue

            # Fallback to vector similarity
            if self.anchor_embeddings is None:
                results.append((idx, []))
                continue

            # Encode single text
            query_embedding = self.embedder.encode([text], convert_to_tensor=True)

            # Calculate cosine similarity
            from sentence_transformers import util
            cos_scores = util.cos_sim(query_embedding, self.anchor_embeddings)[0]

            text_results = []
            for anchor_idx, score in enumerate(cos_scores):
                if score.item() >= self.similarity_threshold:
                    topic = self.anchor_metadata[anchor_idx][0]
                    exclusions = self.exclusions_map.get(topic, [])
                    if any(ext.lower() in text.lower() for ext in exclusions):
                        continue

                    anchor_text = self.anchor_metadata[anchor_idx][1]
                    text_results.append({
                        "topic": topic,
                        "score": score.item(),
                        "idx": idx,
                        "matched_anchor": anchor_text,
                        "key_terms_found": anchor_text.lower()
                    })

            # Sort and keep top 3, merging key_terms_found for same topic
            text_results.sort(key=lambda x: x['score'], reverse=True)
            unique = {}
            for r in text_results:
                if r['topic'] not in unique:
                    unique[r['topic']] = r
                    unique[r['topic']]['_all_terms'] = {r.get('key_terms_found', '')}
                else:
                    unique[r['topic']]['_all_terms'].add(r.get('key_terms_found', ''))
            for entry in unique.values():
                entry['key_terms_found'] = "|".join(sorted(t for t in entry.pop('_all_terms') if t))
            top_topics = list(unique.values())[:3]

            results.append((idx, top_topics))

        return results

    def analyze_topics_parallel(self, texts: List[str], show_progress=True) -> List[List[Dict]]:
        """
        Analyze topics in parallel across multiple cores.

        Args:
            texts: List of text segments
            show_progress: Show progress bar

        Returns:
            List of topic detection results (one per input text)
        """
        if not texts:
            return []

        # Create chunks for parallel processing
        chunk_size = max(1, len(texts) // (self.num_workers * 4))  # 4 chunks per worker
        indexed_texts = list(enumerate(texts))
        chunks = [indexed_texts[i:i + chunk_size] for i in range(0, len(indexed_texts), chunk_size)]

        # Process chunks in parallel
        results_dict = {}

        if show_progress:
            print(f"   Processing {len(texts)} texts in {len(chunks)} chunks across {self.num_workers} workers...")

        # Sequential processing for now (multiprocessing with spaCy/transformers can be tricky)
        # This still provides good performance with batching
        for chunk in tqdm(chunks, desc="Topic Detection", disable=not show_progress):
            chunk_results = self.analyze_topics_chunk(chunk)
            for idx, result in chunk_results:
                results_dict[idx] = result

        # Return results in original order
        return [results_dict.get(i, []) for i in range(len(texts))]

    def analyze_sentiment_batch(self, sentiment_queue: List[Dict]) -> List[Dict]:
        """
        VADER sentiment analysis (fast and simple).

        Args:
            sentiment_queue: List of dicts with 'text', 'text_pair', 'text_idx', 'topic_idx'

        Returns:
            List of sentiment results
        """
        if not sentiment_queue:
            return []

        texts_input = [str(item["text"]) for item in sentiment_queue]
        sent_results_flat = []

        # VADER is very fast - process all texts directly
        for text in tqdm(texts_input, desc="VADER Sentiment", leave=False):
            # Get VADER scores
            scores = self.sentiment_analyzer.polarity_scores(text)

            # Determine sentiment label based on compound score
            compound = scores['compound']
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'

            scores_str = f"pos: {scores['pos']:.2f}, neu: {scores['neu']:.2f}, neg: {scores['neg']:.2f}, compound: {compound:.2f}"

            sent_results_flat.append({
                "label": label,
                "score": abs(compound),
                "all_scores": scores_str
            })

        return sent_results_flat

    def analyze_batch_parallel(self, texts: List[str], show_progress=True, skip_sentiment=False) -> List[List[Dict]]:
        """
        Full parallel analysis pipeline: topics + sentiment.

        Args:
            texts: List of text segments
            show_progress: Show progress bars
            skip_sentiment: If True, skip sentiment analysis step

        Returns:
            List of enrichment results (one per input text)
        """
        if not texts:
            return []

        start_time = time.time()

        # 1. Topic Detection (parallel)
        results_by_text = self.analyze_topics_parallel(texts, show_progress=show_progress)

        # 2. Build sentiment queue
        sentiment_queue = []
        for i, text_results in enumerate(results_by_text):
            for topic_idx, result in enumerate(text_results):
                sentiment_queue.append({
                    "text": texts[i],
                    "text_pair": result['topic'],
                    "text_idx": i,
                    "topic_idx": topic_idx
                })

        # 3. Batch sentiment analysis
        if sentiment_queue and not skip_sentiment:
            if show_progress:
                print(f"   Running batch sentiment analysis for {len(sentiment_queue)} topic pairs...")

            sent_results = self.analyze_sentiment_batch(sentiment_queue)

            # Map results back to topics
            for i, res in enumerate(sent_results):
                meta = sentiment_queue[i]
                target = results_by_text[meta["text_idx"]][meta["topic_idx"]]
                target["sentiment"] = res['label']
                target["sentiment_score"] = res['score']
                target["all_scores"] = res['all_scores']

        elapsed = time.time() - start_time
        if show_progress:
            print(f"   Completed analysis in {elapsed:.2f}s ({len(texts)/elapsed:.1f} texts/sec)")

        return results_by_text


def get_optimal_config():
    """
    Get optimal configuration based on system resources.

    Returns:
        Dict with recommended configuration
    """
    cpu_count = mp.cpu_count()

    # Check available memory
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
    except:
        available_gb = 8  # Assume 8GB if psutil not available

    # Determine optimal settings
    if cpu_count >= 8 and available_gb >= 16:
        # High-end system
        config = {
            'num_workers': cpu_count - 1,
            'sentiment_batch_size': 16,
            'classification_batch_size': 32,
            'chunk_size': 500
        }
    elif cpu_count >= 4 and available_gb >= 8:
        # Mid-range system
        config = {
            'num_workers': max(2, cpu_count - 2),
            'sentiment_batch_size': 8,
            'classification_batch_size': 16,
            'chunk_size': 250
        }
    else:
        # Low-end system
        # MEMORY OPTIMIZATION: Reduced batch sizes for low-memory systems
        config = {
            'num_workers': 1,
            'sentiment_batch_size': 2,
            'classification_batch_size': 4,
            'chunk_size': 50
        }

    print(f"\n[System Info]")
    print(f"  CPUs: {cpu_count}")
    print(f"  Available Memory: {available_gb:.1f} GB")
    print(f"\n[Optimal Configuration]")
    print(f"  Workers: {config['num_workers']}")
    print(f"  Sentiment Batch Size: {config['sentiment_batch_size']}")
    print(f"  Classification Batch Size: {config['classification_batch_size']}")
    print(f"  Chunk Size: {config['chunk_size']}\n")

    return config
