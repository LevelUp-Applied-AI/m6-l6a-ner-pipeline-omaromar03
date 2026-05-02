import pandas as pd
import spacy
from spacy.pipeline import EntityRuler


STANDARD_ENTITIES = { "ORG", "GPE", "DATE", "LAW", "MONEY",
    "PERSON", "QUANTITY", "LOC", "EVENT", "WORK_OF_ART"}
def load_data():
    articles =pd.read_csv("data/climate_articles.csv")
    gold = pd.read_csv("data/gold_entities.csv")
    return articles, gold
def get_climate_patterns():
    return [
        {"label": "CLIMATE_EVENT", "pattern": "COP28"},
        {"label": "CLIMATE_EVENT", "pattern": "COP27"},
        {"label": "AGREEMENT", "pattern": "Paris Agreement"},
        {"label": "REPORT", "pattern": "IPCC AR6"},
        {"label": "REPORT", "pattern": "Sixth Assessment Report"},
        {"label": "ORG", "pattern": "IPCC"},
        {"label": "POLICY", "pattern": "National Climate Change Policy"},
        {"label": "POLICY", "pattern": "Green Deal"},
        {"label": "THRESHOLD", "pattern": "1.5 degrees Celsius"},
        {"label": "THRESHOLD", "pattern": "2°C target"},
        {"label": "THRESHOLD", "pattern": "1.5°C"},
        {"label": "CLIMATE_CONCEPT", "pattern": "net zero"},
        {"label": "CLIMATE_CONCEPT", "pattern": "fossil fuels"},
    ]

def bulid_nlp_with_ruler(position="before"):
    nlp = spacy.load("en_core_web_sm")
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")
    ruler = nlp.add_pipe(
        "entity_ruler",
        before="ner" if position == "before" else None,
        config={"overwrite_ents": True if position == "before" else False},
    )

    ruler.add_patterns(get_climate_patterns())
    return nlp

def extract_entities(df, nlp):
    rows = []

    english_df = df[df["language"] == "en"]

    for _, row in english_df.iterrows():
        doc = nlp(row["text"])

        for ent in doc.ents:
            rows.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })

    return pd.DataFrame(rows)

def compare_counts(base_df,custom_df):
    print("\nBase entity counts:")
    print(base_df["entity_label"].value_counts())

    print("\nCustom entity counts:")
    print(custom_df["entity_label"].value_counts())

    print("\nTotal base entities:", len(base_df))
    print("Total custom entities:", len(custom_df))

def evaluate_standard_only(predicted_df, gold_df):
    predicted_df = predicted_df[predicted_df["entity_label"].isin(STANDARD_ENTITIES)]
    gold_df = gold_df[gold_df["entity_label"].isin(STANDARD_ENTITIES)]

    predicted_set = set(zip(predicted_df["text_id"], predicted_df["entity_text"], predicted_df["entity_label"]))
    gold_set = set(zip(gold_df["text_id"], gold_df["entity_text"], gold_df["entity_label"]))

    tp = len(predicted_set & gold_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
## commit 
    print("\nStandard Entity Evaluation:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

def show_custom_rule_examples(entities_df, articles_df):
    custom_labels = {
        "CLIMATE_EVENT",
        "AGREEMENT",
        "REPORT",
        "POLICY",
        "THRESHOLD",
        "CLIMATE_CONCEPT",
    }

    custom_entities = entities_df[entities_df["entity_label"].isin(custom_labels)]

    print("\nCustom rule examples:")

    for _, ent in custom_entities.head(10).iterrows():
        text = articles_df[articles_df["id"] == ent["text_id"]]["text"].iloc[0]

        print("\nEntity:", ent["entity_text"], "->", ent["entity_label"])
        print("Text:", text[:250], "...")


if __name__ == "__main__":
    articles, gold = load_data()

    base_nlp = spacy.load("en_core_web_sm")
    before_nlp = bulid_nlp_with_ruler(position="before")
    after_nlp = bulid_nlp_with_ruler(position="after")

    base_entities = extract_entities(articles, base_nlp)
    before_entities = extract_entities(articles, before_nlp)
    after_entities = extract_entities(articles, after_nlp)

    print("\n===== BASE vs ENTITYRULER BEFORE NER =====")
    compare_counts(base_entities, before_entities)

    print("\nBase evaluation:")
    print(evaluate_standard_only(base_entities, gold))

    print("\nEntityRuler BEFORE evaluation:")
    print(evaluate_standard_only(before_entities, gold))

    show_custom_rule_examples(before_entities, articles)

    print("\n===== BASE vs ENTITYRULER AFTER NER =====")
    compare_counts(base_entities, after_entities)

    print("\nEntityRuler AFTER evaluation:")
    print(evaluate_standard_only(after_entities, gold))

    show_custom_rule_examples(after_entities, articles)