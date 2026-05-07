# Lab 6A — NER Pipeline

Module 6 Week A lab for AI.SPIRE Applied AI & ML Systems.

In this lab, I built a complete Named Entity Recognition (NER) pipeline using both spaCy and Hugging Face, then compared their outputs and evaluated performance against a gold standard dataset.

---

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

- Hugging Face models run on PyTorch (CPU version to keep size small)
- spaCy model is lightweight (~12MB)
- First run will download the HF model (~250MB) — one-time only

---

## What I Built

The pipeline includes:

- Data loading and exploration  
- Text preprocessing (normalization + lemmatization)  
- Named Entity extraction using two different systems  
- Output comparison between models  
- Evaluation using precision, recall, and F1  

---

## Tasks Implemented

1. **load_data(filepath)**  
   Load the climate dataset into a DataFrame.

2. **explore_data(df)**  
   Generate a summary including:
   - dataset shape  
   - language distribution  
   - category distribution  
   - text length statistics  

3. **preprocess_text(text, nlp)**  
   - Apply Unicode normalization (NFC)  
   - Remove punctuation and spaces  
   - Convert to lowercase lemmas  

4. **extract_spacy_entities(df, nlp)**  
   - Filter English texts  
   - Extract entities using spaCy  
   - Return structured DataFrame  

5. **extract_hf_entities(df, ner_pipeline)**  
   - Use Hugging Face NER pipeline  
   - Merge subword tokens (`##`)  
   - Clean labels (remove `B-` / `I-`)  

6. **compare_ner_outputs(spacy_df, hf_df)**  
   - Count entities per system  
   - Compute overlap:
     - both  
     - spaCy-only  
     - HF-only  

7. **evaluate_ner(predicted_df, gold_df)**  
   - Compute:
     - Precision  
     - Recall  
     - F1 score  
   - Exact match based on:
     `(text_id, entity_text, entity_label)`

---

## Results Summary

- spaCy detected more entities overall and achieved higher recall  
- Hugging Face detected fewer entities and had lower recall  
- Both systems showed very low precision, meaning many predictions were incorrect  

### Key Insight

- spaCy is better at coverage (recall) but introduces noise  
- Hugging Face struggles more, likely due to tokenization (WordPiece splitting)  
- Small gold dataset (~10 texts) makes evaluation unstable  

---

## Example Output

```
spaCy entities: 1202
HF entities: 1125

Both: 230
spaCy-only: 958
HF-only: 826

spaCy F1 ≈ 0.07
HF F1 ≈ 0.01
```

---

## Submission Steps

```bash
git checkout -b lab-6a-ner-pipeline
git add .
git commit -m "Complete NER pipeline lab"
git push --set-upstream origin lab-6a-ner-pipeline
```

Then open a Pull Request to `main` and submit the link on TalentLMS.

---

## Notes

- Results are not expected to be high due to:
  - small gold dataset  
  - strict matching conditions  
- Focus is on correct pipeline implementation and analysis

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.