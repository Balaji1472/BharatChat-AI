**Airavata Quantized Chat App (GGUF-based)**

A simple and efficient chatbot with bilingual support (Hindi & English), powered by AI4Bharat's Airavata LLM in GGUF quantized format.

---

## 🧠 Key Features

* Lightweight FastAPI backend with GGUF inference using `llama-cpp-python`
* Hindi/English language query support
* Streamlit frontend or basic HTML frontend compatible
* Ideal for CPU and low-end GPU inference
* Quantized model (Q4\_K\_M) \~4GB, runs on 8GB RAM machines

---

## 🚀 Tech Stack

* **Backend**: FastAPI (REST API)
* **Inference**: llama-cpp-python (GGUF model loader)
* **Frontend**: Streamlit (or JS-based UI)
* **Model**: `sam749/Airavata-GGUF` (Q4\_K\_M)

---

## 🛠 How It Works

1. User submits prompt (Hindi or English)
2. Backend formats and passes prompt to GGUF model
3. Model generates bilingual response
4. Backend returns result with latency and token info

---

## 📦 Installation

```bash
# Clone the repo
$ git clone https://github.com/Balaji1472/BharatChat-AI.git
$ cd BharatChat-AI

# Setup environment
$ python -m venv venv
$ source venv/bin/activate  # Windows: venv\Scripts\activate
$ pip install -r requirements.txt
```

---

## 📂 Usage

### 1. Place your GGUF model

```bash
# Download and place here:
models/airavata.Q4_K_M.gguf
```

### 2. Start FastAPI backend

```bash
$ python main.py
# or using uvicorn
$ uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Start Streamlit frontend (optional)

```bash
$ streamlit run app.py
```

---

## 📡 API Endpoints

### `POST /generate`

Generates a bilingual response.

```json
{
  "message": "आप कैसे हैं?",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9
}
```

### `GET /health`

Checks if the model is loaded.

---

## 📊 Benchmark Summary (Colab T4)

| Model         | Size (MB) | Latency (ms) | Throughput (tok/s) |
| ------------- | --------- | ------------ | ------------------ |
| GGUF Q4\_K\_M | 3979.2    | 3407         | 23.5               |

---

## 📁 Project Structure

```
.
├── main.py          # FastAPI backend
├── app.py           # Streamlit frontend
├── models/          # Place GGUF model here
├── requirements.txt
└── README.md
```

---

## 📝 License

MIT License. © 2025 Balaji / Open-Source Community

---

## 🙋‍♂️ Acknowledgements

* [AI4Bharat](https://ai4bharat.org/) for Airavata LLM
* [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF format support

---

\*\*Star this repo ⭐ if you found it useful. \*\*
