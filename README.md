# Rewind AI: Emotion-Based Growth Chatbot

A beta version of a web-based AI chatbot designed to help users reflect on emotions and gain self-understanding through empathetic conversation, emotional journaling, and analytical feedback.

## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Built With](#built-with)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## About

**Rewind (Me:Rewind)** is an emotion-driven growth platform that helps users record and understand their feelings, and rebuild relationships. Through conversations with an AI chatbot, users are guided to recognize their own emotional patterns and recurring triggers.

This project tested various LLMs (GPT, Gemini) and TTS/STT tools (CLOVA, Polly, Whisper), resulting in the development of a functional beta version.

---

## Environment Variables

Create a `.env` file in the project root and add your API keys as follows:

```
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

**Do not commit your `.env` file.** It is already included in `.gitignore`.

Share a `.env.example` file with your team, and each member should enter their own values.

---

## Virtual Environment Setup

It is recommended to use a Python virtual environment for dependency management:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Installation

```bash
git clone https://github.com/insunginfo-healthcare/openai-api-test.git
cd openai-api-test
pip install -r requirements.txt
```

---

## Usage

Run the app using Streamlit:

```bash
streamlit run app.py
```

1. Log in or sign up, then select a chatbot persona (Fact-bot, Joy-bot, Sad-bot).
2. Start chatting. Conversations are saved as JSON files.
3. After chatting, you can generate daily and weekly emotional reports.
4. Emotional reports are stored in the main menu.

---

## Features

- 🎭 Emotion-Based Chatbot Personas  
  - Fact-bot: Logical and direct feedback  
  - Sad-bot: Warm empathy and comfort  
  - Joy-bot: Positive feedback and encouragement

- 💬 Context-Aware Conversations  
  - References previous chats and user metadata  
  - Avoids sensitive topics unless the user brings them up

- 📊 Emotional Reports  
  - Daily reports: major emotions, repeated situations, triggers, emotional time zones, behavior patterns, AI interpretation, recommended routines  
  - Weekly summaries generated automatically

- 🔐 User Profile System  
  - Local folder-based login/signup (no backend server)  
  - Personalized chat memory

- 🧠 Prompt Engineering  
  - Role-appropriate phrasing and emotional responses  
  - Adjusts tone based on user intent (e.g., advice requests vs. emotional expression)

- 🗂️ Data Storage  
  - All chat and report data are stored as JSON files (no database used for now)

- 🤖 LLM Models  
  - Uses OpenAI GPT and Google Gemini 2.0 Flash for chatbot responses

- 🚀 Streamlit UI  
  - Provides a web interface for chatbot and emotional report features

---

## Built With

- Streamlit
- Google Gemini 2.0 Flash
- OpenAI GPT
- Pinecone (User Vector Database)

---

## Authors

Seung Jun Hwang – Intern Developer, AI Emotion Platform 
GitHub Repository: [https://github.com/insunginfo-healthcare/openai-api-test](https://github.com/joshuah224/rewind-webapp)

---

## License

This project is licensed for internal educational and prototyping use only. Contact the author for other uses.

---

## Acknowledgments

Guidance and feedback from the Insung Information Healthcare team
Thanks to teammates for UI & UX testing and feedback: Yoochang Lee, Dohyeon Pyeon
