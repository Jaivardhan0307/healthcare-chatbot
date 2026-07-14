\# 🏥 Healthcare Assistant Chatbot



A conversational healthcare assistant built with Python and Streamlit. The bot answers medical FAQs, predicts possible conditions from symptoms, and recommends the right type of doctor — all running live in the browser.



🔗 \*\*Live Demo:\*\* \[Click here to try it](https://healthcare-chatbot-judgsjgbckabnplzkcrwtg.streamlit.app)

\*(Update this link after deployment)\*



\---



\## 🧠 How It Works — Architecture



This chatbot uses a \*\*four-layer response system\*\*. Each user message passes through the layers in order, and the first layer that can answer it wins:

User Message

│

▼

┌─────────────────────────────┐

│  Layer 1: Greeting Handler  │  → Detects "hi", "hello", "hey" etc.

└─────────────────────────────┘

│ (no match)

▼

┌──────────────────────────────────┐

│  Layer 2: Intent Classifier      │  → Detects if user is describing symptoms

└──────────────────────────────────┘

│ symptom intent detected

▼

┌──────────────────────────────────────────────────┐

│  Layer 3: Symptom Checker + Doctor Recommender   │  → Predicts condition, suggests specialist

└──────────────────────────────────────────────────┘

│ (general medical question, not symptoms)

▼

┌────────────────────────────────────────────────────────┐

│  Layer 4a: TF-IDF + Cosine Similarity (FAQ Retrieval)  │  → Searches real medical FAQ dataset

└────────────────────────────────────────────────────────┘

│ (confidence below threshold)

▼

┌───────────────────────────────┐

│  Layer 4b: Flan-T5-base LLM  │  → Generative fallback for anything else

└───────────────────────────────┘



\### Why TF-IDF instead of pure generative AI?

For a healthcare assistant, \*\*accuracy and reliability matter more than creativity\*\*. TF-IDF retrieval finds the closest real answer from a verified medical FAQ dataset — it cannot hallucinate. The generative model (Flan-T5) is only used as a last resort, when no good FAQ match exists. This is a deliberate architectural decision, not a limitation.



\---



\## 🛠️ Tech Stack



| Layer | Tool |

|---|---|

| UI \& App Framework | Streamlit |

| FAQ Retrieval | Scikit-learn (TF-IDF + Cosine Similarity) |

| Language Model | Google Flan-T5-base (via Hugging Face Transformers) |

| Symptom Prediction | Rule-based matching on Kaggle disease-symptom dataset |

| Data Processing | Pandas |

| Language | Python 3 |



\---



\## 📂 Data Sources



All data used in this project comes from real-world public datasets — not handcrafted examples.



| File | Source | Description |

|---|---|---|

| `data/faq.csv` | \[Hugging Face — petkopetkov/medical-question-answering-small](https://huggingface.co/datasets/petkopetkov/medical-question-answering-small) | 500 real medical Q\&A pairs |

| `data/symptom\_disease.csv` | \[Kaggle — itachi9604/disease-symptom-description-dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) | 41 diseases with symptom profiles |

| `data/doctor\_recommendation.csv` | Handcrafted | Specialist type mapped to each disease |



\---



\## 🚀 Run Locally



\*\*Requirements:\*\* Python 3.8+



\*\*1. Clone the repository\*\*

```bash

git clone https://github.com/Jaivardhan0307/healthcare-chatbot.git

cd healthcare-chatbot

```



\*\*2. Install dependencies\*\*

```bash

pip install -r requirements.txt

```



\*\*3. Run the app\*\*

```bash

streamlit run app.py

```



The app will open automatically at `http://localhost:8501`



> ⚠️ Note: The first run downloads the Flan-T5-base model (\~1GB). This is a one-time download and will be cached locally after that.



\---



\## 🗂️ Project Structure

HEALTH\_BOT/

│

├── app.py                  # Main Streamlit app — all four layers live here

├── requirements.txt        # All Python dependencies

│

├── data/

│   ├── faq.csv                    # Medical FAQ dataset (500 rows)

│   ├── symptom\_disease.csv        # Disease-symptom mapping (41 diseases)

│   └── doctor\_recommendation.csv  # Disease-to-specialist mapping

│

└── README.md



\---



\## 🏗️ How It Was Built — Stage by Stage



This project was built incrementally, one feature at a time, the way real software is developed:



| Stage | What was built |

|---|---|

| Stage 1 | Basic chatbot with Flan-T5-base, chat bubble UI, session memory, medical disclaimer |

| Stage 2 | Real-world data integration, TF-IDF + cosine similarity retrieval layer |

| Stage 3 | Intent classification, symptom prediction, doctor recommendation |

| Stage 4 | Sidebar, chat history export to JSON, requirements.txt, threshold tuning |

| Stage 5 | Deployment to Streamlit Cloud, README, merge to main |



\---



\## ⚠️ Disclaimer



This chatbot is for \*\*educational and informational purposes only\*\*. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.



\---



\## 👩‍💻 Author



\*\*Jahnavi\*\* — Final year student  

GitHub: \[@Jaivardhan0307](https://github.com/Jaivardhan0307)

