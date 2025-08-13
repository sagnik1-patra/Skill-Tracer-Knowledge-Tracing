SkillTracer — Knowledge Tracing & Next-Item Recommender
A compact, production-ready pipeline to train, evaluate, and serve a student knowledge-tracing model (DKT/LSTM) with a next-item recommender.
Everything in this folder is designed to run locally on CPU and to deploy as a small FastAPI service.

Base folder:
C:\Users\sagni\Downloads\SkillTracer Knowledge Tracing

Contents
csharp
Copy
Edit
SkillTracer Knowledge Tracing/
├─ archive (1)/2012-2013-data-with-predictions-4-final.csv    # dataset (ASSISTments-style)
├─ model.keras | model.h5                                      # trained model (prefer .keras)
├─ preprocessor.pkl                                            # skill maps + token rules
├─ threshold.json                                              # best F1 decision threshold
├─ metrics.json                                                # AUC/AP/Brier, etc.
├─ accuracy_plot.png | roc_curve.png | pr_curve.png
├─ calibration.png | confusion_matrix.png
├─ predictor.py                                                # inference helpers (used by API)
├─ app.py                                                      # /predict API
├─ recommender.py                                              # mastery + 1PL difficulty recommender
├─ reco_catalog.pkl | reco_catalog_preview.json                # stats for recommender
├─ app2.py                                                     # /predict + /recommend API
├─ index.html                                                  # tiny browser UI
├─ requirements.txt | Dockerfile | Procfile | run_local.bat
├─ sample_history.json
└─ (optional) predict_cli.py                                   # local CLI
Quick start (local)
Open a terminal:

powershell
Copy
Edit
cd "C:\Users\sagni\Downloads\SkillTracer Knowledge Tracing"
python -m pip install --upgrade pip
pip install -r requirements.txt
Start the API with prediction and recommendation:

powershell
Copy
Edit
uvicorn app2:app --host 0.0.0.0 --port 8000
Try it:

Swagger/Docs: http://localhost:8000/docs

Minimal UI: open index.html in your browser

Health check:

bash
Copy
Edit
curl http://localhost:8000/health
What the model does
Knowledge tracing (DKT/LSTM): predicts the probability the student’s next response is correct based on a sequence of (skill, correct) interactions.

Next-item recommendation: estimates per-skill mastery from recent history (EMA) and recommends problems in a target success band (default 60–75%) using a simple 1PL/IRT-style difficulty signal learned from historical correctness rates.

Artifacts & plots
model.keras (preferred) or model.h5 (legacy)

preprocessor.pkl (skill → id mapping, tokenization rules, max_len)

threshold.json (best F1 threshold tuned on held-out data)

metrics.json (AUC, AP, Brier, best threshold, …)

Training curves: accuracy_plot.png

Ranking quality: roc_curve.png, pr_curve.png

Calibration: calibration.png

Confusion matrix (at tuned threshold): confusion_matrix.png

Confusion Matrix (example)


API endpoints
GET /health
Returns model and catalog info.

Example

json
Copy
Edit
{
  "status": "ok",
  "model": {
    "n_skills": 123,
    "max_len": 200,
    "token_vocab_size": 247,
    "uses_threshold_file": true,
    "has_model_keras": true,
    "has_model_h5": false
  }
}
POST /predict
Predict the probability that the next answer is correct.

Request

json
Copy
Edit
{
  "history": [
    {"skill":"Algebra","correct":1},
    {"skill":"Algebra","correct":0},
    {"skill":"Fractions","correct":1}
  ],
  "threshold": 0.5
}
You can also send pairs like ["Algebra", 1].

Response

json
Copy
Edit
{
  "probability": 0.73,
  "threshold": 0.5,
  "predicted_class": 1
}
POST /recommend
Recommend the next best problems (or skills if item IDs aren’t present) targeting a success band.

Request

json
Copy
Edit
{
  "history": [["Algebra",1],["Algebra",0],["Fractions",1]],
  "top_k": 5,
  "target_low": 0.60,
  "target_high": 0.75,
  "min_item_count": 30
}
Response (item-level; falls back to skills)

json
Copy
Edit
{
  "recommendations": [
    {
      "problem_id": "12345",
      "skill": "Fractions",
      "pred_success": 0.68,
      "seen": 412,
      "p_item": 0.64,
      "difficulty_b": 0.59,
      "score": -0.02
    }
  ]
}
Minimal HTML UI
Open index.html, paste a JSON history, and click Predict or Recommend.
The page assumes the API is running on the same origin (/predict, /recommend).

How it works
Tokenization (DKT)
For each timestep t, build token:
token = 1 + skill_id + correctness * n_skills
(0 is PAD; vocab size 2*n_skills + 1)

Model: Embedding → LSTM(return_sequences=True) → Dropout → Dense(sigmoid)

Mastery & Recommendation
Mastery (per skill) via EMA on correctness:
m_new = (1−decay)*m_prev + decay*correct (default decay = 0.3)

Catalog (reco_catalog.pkl): per-skill & per-item p_correct with Laplace smoothing (k=1), plus 1PL difficulty b = −logit(p_item).

Item success prediction:
P(correct) = sigmoid(θ_student − b_item) where
θ_student = logit(0.9 * mastery + 0.1 * global_skill_p)

Selection: choose items with P(correct) in [0.60, 0.75] (configurable).

Training / evaluation (notebook cells)
The project includes notebook cells that:

Train the DKT model and save:

preprocessor.pkl, label_encoder.pkl

model.h5 and model.keras

model_config.yaml (or .json if PyYAML not installed)

Evaluate and save:

accuracy_plot.png, roc_curve.png, pr_curve.png, calibration.png

confusion_matrix.png, metrics.json, threshold.json

Tip: Always save native Keras format to avoid legacy H5 issues:

python
Copy
Edit
model.save(r"C:\Users\sagni\Downloads\SkillTracer Knowledge Tracing\model.keras")
Dataset schema (auto-detected)
The loader tries to detect these columns:

Student: student_id, user_id, Anon Student Id, student, sid

Skill: skill_id, skill, tag, KC(SubSkills), KC, skill_name, concept_id

Problem (optional): problem_id, item_id, question_id, Step ID, Problem Name

Label: correct, is_correct, Correct First Attempt, answered_correctly, label

If multiple skills appear in a cell, it uses the first.
