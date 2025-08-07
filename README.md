# Rewind AI: 감정 기반 성장 챗봇

A beta version of a web-based AI chatbot designed to help users reflect on emotions and gain self-understanding through empathetic conversation, emotional journaling, and analytical feedback.

## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Features](#features)
- [Built With](#built-with)
- [Contributing](#contributing)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## About

**Rewind (Me:Rewind)** is an emotion-driven growth platform that helps users reflect on their feelings, understand themselves, and rebuild relationships. Built using AI chatbot interactions, the system encourages users to identify their own patterns and emotional triggers through guided conversation and personalized emotional reports.

During this project, various LLMs (GPT, Gemini) and personalization strategies were tested to build a functional beta version.

---

## Getting Started

### Prerequisites

- Python 3.9+
- Streamlit
- OpenAI / Google Generative AI credentials
- Pinecone API (optional, vector DB not used in final storage)

### Installation

```bash
git clone https://github.com/insunginfo-healthcare/openai-api-test.git
cd openai-api-test
pip install -r requirements.txt
streamlit run app.py
```

## Usage
Run the app using Streamlit.

Choose a chatbot persona: 팩폭이 (Anger), 우울이 (Sadness), or 기쁨이 (Joy).

Start chatting. The chatbot personalizes responses based on your metadata.

Optionally save the conversation at the end to generate a daily emotional report.

Weekly reports are automatically generated every Monday using the last 7 days’ data.

## Features
- 🎭 Emotion-Aware Chatbot Personas
  - 팩폭이: Logical and direct advice
  - 우울이: Warm empathy and comforting tone
  - 기쁨이: Positive reinforcement and encouragement

- 💬 Context-Aware Conversations
  - Previous chats and user metadata are referenced
  - Sensitive topics avoided unless user brings them up

- 📊 Emotional Reports
  - Daily reports include: major emotions, repeated situations, triggers, emotional time zones, behavior patterns, and AI interpretations
  - Weekly summaries generated automatically

- 🔐 User Profile System
  - Login/Signup via local folder management (no backend server)
  - Personalized chat memory

- 🧠 Prompt Engineering
  - Role-appropriate phrasing and emotional responses
  - Adjusts tone based on user intent (e.g., asking for advice vs. expressing frustration)

## Built With
- Streamlit
- Google Gemini 2.0 Flash
- OpenAI GPT
- Pinecone (for vector memory testing)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

## Authors
Joshua (Seung Jun) Hwang (황승준) – Intern Developer
[GitHub Repository]([url](https://github.com/insunginfo-healthcare/openai-api-test))

## License
This project is licensed for internal educational and prototyping use. Contact the author for other uses.

## Acknowledgments
Internship guided by [insunginfo-healthcare team]

Special thanks to my teammates for feedback: Lee Yoochang, Pyeon Do-Heon
