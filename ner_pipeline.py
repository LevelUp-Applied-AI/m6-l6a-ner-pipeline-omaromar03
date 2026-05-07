"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

import unicodedata

import pandas as pd
import numpy as np
import spacy
from transformers import pipeline as hf_pipeline
import matplotlib.pyplot as plt

def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    # TODO: Load the CSV and return the DataFrame
    return pd.read_csv(filepath)


def explore_data(df):
    """Summarize basic corpus statistics.

    Args:
        df: DataFrame returned by load_data.

    Returns:
        Dictionary with keys:
          'shape': tuple (n_rows, n_cols)
          'lang_counts': dict mapping language code -> row count
          'category_counts': dict mapping category -> row count
          'text_length_stats': dict with 'mean', 'min', 'max' word counts
    """
    # TODO: Compute shape, language/category value_counts, and word-count
    #       statistics on df['text']
    text_lengths = df['text'].apply(lambda x: len(str(x).split()))
    return {
        'shape': df.shape,
        'lang_counts': df['language'].value_counts().to_dict(),
        'category_counts': df['category'].value_counts().to_dict(),
        'text_length_stats': {
            'mean': text_lengths.mean(),
            'min': text_lengths.min(),
            'max': text_lengths.max()
        },
    }


def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using the injected spaCy pipeline.

    Args:
        text: Raw text string.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        List of cleaned, lemmatized token strings.
    """
    # TODO: NFC-normalize the text, run it through nlp(), drop
    #       punctuation/whitespace tokens, return lowercased lemmas
    normalized_text = unicodedata.normalize('NFC', text)
    doc = nlp(normalized_text)
    tokens = []
    for token in doc:
        if not token.is_punct and not token.is_space:
            tokens.append(token.lemma_.lower())
    return tokens




def extract_spacy_entities(df, nlp):
    """Extract named entities from English texts using spaCy NER.

    Args:
        df: DataFrame with columns id, text, language, ...
        nlp: A loaded spaCy Language object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # TODO: Filter df to English rows, process each text with nlp,
    #       collect entities into rows, return as a DataFrame
    rows = []
    english_df = df[df['language'] == 'en']
    for _, row in english_df.iterrows():
        text_id = row['id']
        text = row['text']
        doc = nlp(text)
        for ent in doc.ents:
            rows.append({
                'text_id': text_id,
                'entity_text': ent.text,
                'entity_label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char
            })
    return pd.DataFrame(rows)

def merge_hf_entities(raw_entities):
    merged = []

    current_entity = None

    for ent in raw_entities:
        word = ent["word"]
        label = ent["entity"]

        clean_label = label.replace("B-", "").replace("I-", "")

        if word.startswith("##") and current_entity is not None:
            current_entity["entity_text"] += word.replace("##", "")
            current_entity["end_char"] = ent["end"]
        else:
            if current_entity is not None:
                merged.append(current_entity)

            current_entity = {
                "entity_text": word,
                "entity_label": clean_label,
                "start_char": ent["start"],
                "end_char": ent["end"],
            }

    if current_entity is not None:
        merged.append(current_entity)

    return merged

def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER.

    Uses the injected HF pipeline (expected: dslim/bert-base-NER).

    Args:
        df: DataFrame with columns id, text, language, ...
        ner_pipeline: A loaded Hugging Face `pipeline('ner', ...)` object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # TODO: Filter df to English rows, run each text through
    #       ner_pipeline, merge ## subword tokens, strip B-/I- prefix
    #       from labels (IOB format), return as a DataFrame
    rows = []
    english_df = df[df['language'] == 'en']
    for _, row in english_df.iterrows():
        text_id = row['id']
        text = row['text']
        raw_entities = ner_pipeline(text)
        merged_entities = merge_hf_entities(raw_entities)
        for ent in merged_entities:
            rows.append({
                'text_id': text_id,
                'entity_text': ent['entity_text'],
                'entity_label': ent['entity_label'],
                'start_char': ent['start_char'],
                'end_char': ent['end_char']
            })
    return pd.DataFrame(rows)


def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label -> count for spaCy
          'hf_counts': dict of entity_label -> count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
          'both': set of (text_id, entity_text) tuples found by both systems
          'spacy_only': set of (text_id, entity_text) tuples found only by spaCy
          'hf_only': set of (text_id, entity_text) tuples found only by HF
    """
    # TODO: Count entities per label for each system, compute totals,
    #       and derive the three overlap sets by matching on
    #       (text_id, entity_text)
    spacy_counts = spacy_df['entity_label'].value_counts().to_dict()
    hf_counts = hf_df['entity_label'].value_counts().to_dict()
    spacy_set = set(zip(spacy_df['text_id'], spacy_df['entity_text']))
    hf_set = set(zip(hf_df['text_id'], hf_df['entity_text']))
    both = spacy_set.intersection(hf_set)
    spacy_only = spacy_set.difference(hf_set)
    hf_only = hf_set.difference(spacy_set)

    return {
        'spacy_counts': spacy_counts,
        'hf_counts': hf_counts,
        'total_spacy': len(spacy_df),
        'total_hf': len(hf_df),
        'both': both,
        'spacy_only': spacy_only,
        'hf_only': hf_only
    }



def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations.

    Computes entity-level precision, recall, and F1. An entity is a
    true positive if both the entity text and label match a gold entry
    for the same text_id.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text,
                      entity_label.
        gold_df: DataFrame with columns text_id, entity_text,
                 entity_label.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1' (floats 0-1).
    """
    # TODO: Match predicted entities to gold entities by text_id +
    #       entity_text + entity_label, compute precision/recall/F1
    predicted_set = set(zip(predicted_df['text_id'], predicted_df['entity_text'], predicted_df['entity_label']))
    gold_set = set(zip(gold_df['text_id'], gold_df['entity_text'], gold_df['entity_label']))
    true_positives = len(predicted_set & gold_set)
    false_positives = len(predicted_set - gold_set)
    false_negatives = len(gold_set - predicted_set)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {   
         'precision': precision,
         'recall': recall,   
         'f1': f1
 }
def entity_counts_by_category(entities_df, category_df):
    """Count entities by category based on text_id mapping.

    Args:
        entities_df: DataFrame with columns text_id, entity_text,
                     entity_label.
        category_df: Original DataFrame with columns id, category.

    Returns:
        Dictionary mapping category -> dict of entity_label -> count.
    """
    marged_df = entities_df.merge(category_df[['id', 'category']], 
                                  left_on='text_id',
                                    right_on='id',
                                      how='left')
    counts = (
        marged_df.groupby(['category', 'entity_label']).size().unstack(fill_value=0)
        )
    return counts 
def evaluate_by_category(predicted_df, gold_df, category_df):
    """Evaluate NER performance by category.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text,
                      entity_label.
        gold_df: DataFrame with columns text_id, entity_text,
                 entity_label.
        category_df: Original DataFrame with columns id, category.

    Returns:
        Dictionary mapping category -> dict with keys 'precision',
        'recall', 'f1' for that category.
    """
    gold_with_category = gold_df.merge(category_df[['id', 'category']],
                                      left_on='text_id',
                                      right_on='id',
                                      how='left')
    results = {}
    for category in gold_with_category['category'].dropna().unique():
        category_gold = gold_with_category[gold_with_category['category'] == category]
        category_predicted = predicted_df[predicted_df['text_id'].isin(category_gold['text_id'])]
        metrics = evaluate_ner(category_predicted, category_gold)
        results[category] = metrics 
    return results
def plot_entity_counts_by_category(counts_df,output_path="entity_counts_by_category.png"):
    """Plot entity counts by category as a stacked bar chart.

    Args:
        counts_df: DataFrame with categories as index and entity labels as columns.
        output_path: File path to save the plot image.
    """

    plt.figure(figsize=(10, 6))
    plt.imshow(counts_df, aspect='auto')
    plt.xticks(range(len(counts_df.columns)), counts_df.columns, rotation=45)
    plt.yticks(range(len(counts_df.index)), counts_df.index)    
    plt.colorbar(label='Entity Count')
    plt.title('Entity Counts by Category')
    plt.xlabel('Entity Label')
    plt.ylabel('Category')
    for i in range(counts_df.shape[0]):
        for j in range(counts_df.shape[1]):
            plt.text(j, i, counts_df.iloc[i, j], ha='center', va='center')


    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
def normalize_entity_label(entity_text):
    entity_map={
        "UN":"United Nations",
        "IPCC":"Intergovernmental Panel on Climate Change",
        "United Nations": "United Nations",
        "IPCC": "IPCC",
        "COP28": "COP28",
        "UAE": "United Arab Emirates",
        "United Arab Emirates": "United Arab Emirates",
    }
    clean_text = str(entity_text).strip()
    return entity_map.get(clean_text, clean_text)
def normalize_entities(entities_df):
    """Normalize entity labels to a consistent set.

    Args:
        entity_df: DataFrame with columns text_id, entity_text,
                   entity_label.

    Returns:
        DataFrame with normalized entity_label values.
    """
    normalize_df = entities_df.copy()
    normalize_df['entity_label'] = normalize_df['entity_text'].apply(normalize_entity_label)
    return normalize_df

def entity_overlap(span1, span2):
    start1, end1 = span1
    start2, end2 = span2
    return max(start1, start2) < min(end1, end2)


def build_entity_spans(df):
    return [
        (row["text_id"], row["entity_text"], row["entity_label"],
         row["start_char"], row["end_char"])
        for _, row in df.iterrows()
    ]


def evaluate_ner_advanced(predicted_df, gold_df):
    predicted = build_entity_spans(predicted_df)
    gold = build_entity_spans(gold_df)

    tp_exact = 0
    tp_partial = 0
    tp_type_agnostic = 0

    boundary_errors = 0
    type_errors = 0
    missing = 0
    spurious = 0

    matched_pred = set()
    matched_gold = set()

    for i, g in enumerate(gold):
        g_id, g_text, g_label, g_start, g_end = g
        found_match = False

        for j, p in enumerate(predicted):
            p_id, p_text, p_label, p_start, p_end = p

            if g_id != p_id:
                continue

            # Exact match
            if (g_text == p_text) and (g_label == p_label):
                tp_exact += 1
                matched_pred.add(j)
                matched_gold.add(i)
                found_match = True
                break

            # Partial match (overlap)
            if entity_overlap((g_start, g_end), (p_start, p_end)):
                if g_label == p_label:
                    tp_partial += 1
                else:
                    type_errors += 1

                tp_type_agnostic += 1
                matched_pred.add(j)
                matched_gold.add(i)
                found_match = True
                break

        if not found_match:
            missing += 1

    for j in range(len(predicted)):
        if j not in matched_pred:
            spurious += 1

    precision = tp_exact / (tp_exact + spurious) if (tp_exact + spurious) > 0 else 0
    recall = tp_exact / (tp_exact + missing) if (tp_exact + missing) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision_exact": precision,
        "recall_exact": recall,
        "f1_exact": f1,
        "tp_exact": tp_exact,
        "tp_partial": tp_partial,
        "tp_type_agnostic": tp_type_agnostic,
        "errors": {
            "missing": missing,
            "spurious": spurious,
            "type_errors": type_errors
        }
    }

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")

    df = load_data()

    summary = explore_data(df)
    print(f"Shape: {summary['shape']}")
    print(f"Languages: {summary['lang_counts']}")
    print(f"Categories: {summary['category_counts']}")
    print(f"Text length (words): {summary['text_length_stats']}")

    print(df[df['language'] == 'en'])
    sample_row = df[df["language"] == "en"].iloc[0]
    sample_tokens = preprocess_text(sample_row["text"], nlp)
    print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")

    print("\nInspecting one spaCy example:")
    doc = nlp(sample_row["text"])
    for ent in doc.ents:
        print(ent.text, "->", ent.label_)

    spacy_entities = extract_spacy_entities(df, nlp)
    print(f"\nspaCy entities: {len(spacy_entities)} total")

    hf_entities = extract_hf_entities(df, hf_ner)
    print(f"HF entities: {len(hf_entities)} total")

    comparison = compare_ner_outputs(spacy_entities, hf_entities)
    print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
    print(f"spaCy-only: {len(comparison['spacy_only'])}")
    print(f"HF-only: {len(comparison['hf_only'])}")

    gold = pd.read_csv("data/gold_entities.csv")

    spacy_metrics = evaluate_ner(spacy_entities, gold)
    print(f"\nspaCy evaluation: {spacy_metrics}")

    hf_metrics = evaluate_ner(hf_entities, gold)
    print(f"Hugging Face evaluation: {hf_metrics}")

    print("\nTier 1 Challenge: Entity Counts by Category")

    spacy_category_counts = entity_counts_by_category(spacy_entities, df)
    print("\nspaCy entity counts by category:")
    print(spacy_category_counts)

    hf_category_counts = entity_counts_by_category(hf_entities, df)
    print("\nHugging Face entity counts by category:")
    print(hf_category_counts)

    print("\nspaCy evaluation by category:")
    spacy_category_metrics = evaluate_by_category(spacy_entities, gold, df)
    print(spacy_category_metrics)

    print("\nHugging Face evaluation by category:")
    hf_category_metrics = evaluate_by_category(hf_entities, gold, df)
    print(hf_category_metrics)

    plot_entity_counts_by_category(
        spacy_category_counts,
        "spacy_entity_counts_by_category.png"
    )

    plot_entity_counts_by_category(
        hf_category_counts,
        "hf_entity_counts_by_category.png"
    )

    print("\nTier 2 Challenge: Entity Normalization")

    normalized_spacy_entities = normalize_entities(spacy_entities)
    print("\nNormalized spaCy entities sample:")
    print(normalized_spacy_entities.head())

    normalized_hf_entities = normalize_entities(hf_entities)
    print("\nNormalized Hugging Face entities sample:")
    print(normalized_hf_entities.head())

    print("\nTier 3 Challenge: Advanced NER Evaluation")

    spacy_adv = evaluate_ner_advanced(spacy_entities, gold)
    print("\nspaCy advanced evaluation:")
    print(spacy_adv)

    hf_adv = evaluate_ner_advanced(hf_entities, gold)
    print("\nHugging Face advanced evaluation:")
    print(hf_adv)