# Stretch Analysis — Custom NER with EntityRuler

## Overview

In this stretch task, I extended the spaCy Named Entity Recognition (NER) pipeline by adding an EntityRuler with custom rules tailored to climate-related text. The goal was to improve entity detection for domain-specific terms that the base model struggles to recognize.

---

## What I Implemented

I created an EntityRuler with custom patterns for climate-related concepts, including:

- Climate events: COP28, COP27  
- Agreements: Paris Agreement  
- Reports: IPCC AR6, Sixth Assessment Report  
- Policies: National Climate Change Policy  
- Thresholds: 1.5°C, 2°C target  
- Concepts: fossil fuels, net zero  

I tested two configurations:

1. EntityRuler **before** the NER model  
2. EntityRuler **after** the NER model  

---

## Results Comparison

### Base Model
- Total entities: 1202  
- F1 score: ~0.091  

### EntityRuler BEFORE NER
- Total entities: 1212  
- F1 score: ~0.092  

### EntityRuler AFTER NER
- Total entities: 1211  
- F1 score: ~0.091  

---

## Key Observations

### 1. Improved Coverage (Recall)

The total number of detected entities increased after adding the EntityRuler. This indicates that the system is now capturing more entities, especially domain-specific ones.

Examples:
- COP28 → CLIMATE_EVENT  
- Paris Agreement → AGREEMENT  
- fossil fuels → CLIMATE_CONCEPT  

---

### 2. Minimal Impact on F1 Score

The F1 score improved slightly when using EntityRuler before the NER model. However, the improvement was small.

Reason:
- The gold dataset only includes standard entity labels (ORG, GPE, DATE, etc.)
- Custom labels (e.g., CLIMATE_EVENT) are not part of the evaluation

---

### 3. BEFORE vs AFTER Behavior

- EntityRuler BEFORE NER slightly improved performance  
- EntityRuler AFTER NER had almost no effect  

Interpretation:
- When placed before NER, rules can guide or override the model  
- When placed after NER, rules act more as corrections  

---

### 4. Domain Knowledge Helps

The base spaCy model missed important climate-specific entities such as:
- COP28  
- Paris Agreement  
- National Climate Change Policy  

After adding rules, these entities were correctly detected.

---

## Limitations

- The evaluation does not reflect the value of custom entities because they are not included in the gold dataset  
- Some rules may overwrite correct predictions if not carefully designed  
- Rule-based systems require manual effort and domain knowledge  

---

## Conclusion

Adding an EntityRuler improved the system's ability to detect domain-specific entities and slightly increased overall performance. While the improvement in F1 score was small, the qualitative results show that combining rule-based and machine learning approaches creates a more robust NER system.

This demonstrates the importance of domain knowledge in real-world NLP applications.